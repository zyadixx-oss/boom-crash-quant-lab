#property strict
#property version "1.00"
#property description "Read-only bounded quote export; no execution or historical cost verification."

// Input dates are UTC epoch values; no local/server-time conversion is applied.
input string   InpSymbol = "Boom 500 Index";
input datetime InpFromDate = D'2025.11.01 00:00:00';
input datetime InpEndDate = D'2025.11.01 00:01:00'; // Exclusive.
input string   InpPrefix = "ResearchQuotes_Boom500_20251101_60s";
input int      InpConnectionWaitSeconds = 30;

string g_server = "";
string g_company = "";
string g_account_mode = "UNKNOWN";
string g_status = "FAILED_INPUT";
string g_reason = "";
bool   g_context_valid = false;
bool   g_symbol_eligible = false;
string g_symbol_custom_observed = "NOT_CHECKED";
string g_spec_status = "NOT_CAPTURED";
string g_spec_capture_time = "NOT_CAPTURED";
string g_spec_keys[];
string g_spec_values[];
bool   g_spec_buffer_ok = true;
bool   g_terminal_trade_allowed = false;
bool   g_mql_trade_allowed = false;
bool   g_api_returned_without_error = false;
int    g_guard_error = 0;
int    g_last_error = 0;
int    g_attempts = 0;
int    g_selected_attempt = 0;
int    g_copy_returned_count = -1;
int    g_attempt_errors[3];
int    g_attempt_counts[3];
int    g_attempt_array_rows[3];
ulong  g_attempt_elapsed_ms[3];
ulong  g_connection_wait_ms = 0;
int    g_write_error = 0;
bool   g_write_ok = true;

string CleanValue(string value)
  {
   StringReplace(value,"\t"," ");
   StringReplace(value,"\r"," ");
   StringReplace(value,"\n"," ");
   return value;
  }

string BoolText(const bool value) { return value ? "true" : "false"; }
string IntText(const long value) { return IntegerToString(value); }
string ULongText(const ulong value) { return StringFormat("%I64u",value); }
string RealText(const double value) { return StringFormat("%.17g",value); }
string GmtText(const datetime value)
  {
   return TimeToString(value,TIME_DATE|TIME_SECONDS)+" UTC";
  }

bool SafePrefix(const string value)
  {
   int length=StringLen(value);
   if(length<1 || length>64) return false;
   for(int i=0;i<length;i++)
     {
      ushort c=StringGetCharacter(value,i);
      bool letter=(c>='a' && c<='z') || (c>='A' && c<='Z');
      bool digit=(c>='0' && c<='9');
      if(!letter && !digit && c!='_' && c!='-') return false;
     }
   string lower=value;
   StringToLower(lower);
   if(lower=="con" || lower=="prn" || lower=="aux" || lower=="nul") return false;
   if(StringLen(lower)==4 && (StringSubstr(lower,0,3)=="com" || StringSubstr(lower,0,3)=="lpt"))
     {
      ushort last=StringGetCharacter(lower,3);
      if(last>='1' && last<='9') return false;
     }
   return true;
  }

bool TradingDisabled()
  {
   ResetLastError();
   long terminal_allowed=TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   int terminal_error=GetLastError();
   ResetLastError();
   long program_allowed=MQLInfoInteger(MQL_TRADE_ALLOWED);
   int program_error=GetLastError();
   g_terminal_trade_allowed=(terminal_allowed!=0);
   g_mql_trade_allowed=(program_allowed!=0);
   g_guard_error=(terminal_error!=0 ? terminal_error : program_error);
   return terminal_error==0 && program_error==0 && terminal_allowed==0 && program_allowed==0;
  }

bool Connected()
  {
   ResetLastError();
   long connected=TerminalInfoInteger(TERMINAL_CONNECTED);
   int error=GetLastError();
   if(error!=0) g_guard_error=error;
   return error==0 && connected==1;
  }

bool WaitForExistingConnection()
  {
   ulong started=GetTickCount64();
   ulong limit=(ulong)InpConnectionWaitSeconds*1000;
   while(true)
     {
      if(!TradingDisabled())
        {
         g_status="FAILED_UNSAFE";
         g_reason="terminal_or_script_trading_not_disabled_or_guard_unavailable";
         g_connection_wait_ms=GetTickCount64()-started;
         return false;
        }
      if(Connected())
        {
         g_connection_wait_ms=GetTickCount64()-started;
         return true;
        }
      g_connection_wait_ms=GetTickCount64()-started;
      if(IsStopped() || g_connection_wait_ms>=limit) break;
      Sleep(100);
     }
   g_status="FAILED_DISCONNECTED";
   g_reason="existing_terminal_connection_unavailable_after_bounded_wait";
   return false;
  }

bool ReadContext(string &server,string &company,string &mode)
  {
   if(!TradingDisabled())
     {
      g_status="FAILED_UNSAFE";
      g_reason="terminal_or_script_trading_not_disabled_or_guard_unavailable";
      return false;
     }
   if(!Connected())
     {
      g_status="FAILED_DISCONNECTED";
      g_reason="terminal_disconnected";
      return false;
     }
   ResetLastError();
   server=AccountInfoString(ACCOUNT_SERVER);
   int server_error=GetLastError();
   ResetLastError();
   company=AccountInfoString(ACCOUNT_COMPANY);
   int company_error=GetLastError();
   ResetLastError();
   long account_mode=AccountInfoInteger(ACCOUNT_TRADE_MODE);
   int mode_error=GetLastError();
   string lower=server;
   StringToLower(lower);
   if(server_error!=0 || company_error!=0 || mode_error!=0 || StringFind(lower,"deriv")!=0)
     {
      g_status="FAILED_ACCOUNT_CONTEXT";
      g_reason="unavailable_or_non_deriv_server_context";
      return false;
     }
   if(account_mode==ACCOUNT_TRADE_MODE_DEMO) mode="DEMO";
   else if(account_mode==ACCOUNT_TRADE_MODE_REAL) mode="LIVE";
   else
     {
      g_status="FAILED_ACCOUNT_CONTEXT";
      g_reason="unsupported_account_mode";
      return false;
     }
   return true;
  }

bool SameContext()
  {
   string server,company,mode;
   if(!ReadContext(server,company,mode)) return false;
   if(server!=g_server || company!=g_company || mode!=g_account_mode)
     {
      g_status="FAILED_ACCOUNT_CONTEXT";
      g_reason="server_company_or_account_mode_changed_during_export";
      return false;
     }
   return true;
  }

void RecordWrite(const uint written)
  {
   if(written==0)
     {
      g_write_ok=false;
      if(g_write_error==0) g_write_error=GetLastError();
     }
  }

void Put(const int handle,const string field,const string value)
  {
   ResetLastError();
   RecordWrite(FileWrite(handle,field,CleanValue(value)));
  }

void BufferSpec(const string key,const string value)
  {
   if(!g_spec_buffer_ok) return;
   int size=ArraySize(g_spec_keys);
   if(ArrayResize(g_spec_keys,size+1)!=size+1 || ArrayResize(g_spec_values,size+1)!=size+1)
     {
      g_spec_buffer_ok=false;
      ArrayFree(g_spec_keys);
      ArrayFree(g_spec_values);
      return;
     }
   g_spec_keys[size]=key;
   g_spec_values[size]=value;
  }

void CaptureDoubleProperty(const string key,const ENUM_SYMBOL_INFO_DOUBLE property)
  {
   double value=0.0;
   ResetLastError();
   bool available=SymbolInfoDouble(InpSymbol,property,value);
   int error=GetLastError();
   BufferSpec(key,available && MathIsValidNumber(value) ? RealText(value) : "NOT_AVAILABLE");
   BufferSpec(key+"_error_code",IntText(error));
  }

void CaptureIntegerProperty(const string key,const ENUM_SYMBOL_INFO_INTEGER property)
  {
   long value=0;
   ResetLastError();
   bool available=SymbolInfoInteger(InpSymbol,property,value);
   int error=GetLastError();
   BufferSpec(key,available ? IntText(value) : "NOT_AVAILABLE");
   BufferSpec(key+"_error_code",IntText(error));
  }

void CaptureCurrentSpec()
  {
   if(!g_context_valid || !g_symbol_eligible) return;
   if(!SameContext())
     {
      g_context_valid=false;
      g_spec_status="UNSTABLE_NOT_CAPTURED";
      return;
     }
   string capture_time=GmtText(TimeGMT());
   CaptureDoubleProperty("symbol_point",SYMBOL_POINT);
   CaptureIntegerProperty("symbol_digits",SYMBOL_DIGITS);
   CaptureDoubleProperty("symbol_trade_contract_size",SYMBOL_TRADE_CONTRACT_SIZE);
   CaptureDoubleProperty("symbol_trade_tick_size",SYMBOL_TRADE_TICK_SIZE);
   CaptureDoubleProperty("symbol_trade_tick_value",SYMBOL_TRADE_TICK_VALUE);
   CaptureDoubleProperty("symbol_trade_tick_value_profit",SYMBOL_TRADE_TICK_VALUE_PROFIT);
   CaptureDoubleProperty("symbol_trade_tick_value_loss",SYMBOL_TRADE_TICK_VALUE_LOSS);
   CaptureDoubleProperty("symbol_volume_min",SYMBOL_VOLUME_MIN);
   CaptureDoubleProperty("symbol_volume_max",SYMBOL_VOLUME_MAX);
   CaptureDoubleProperty("symbol_volume_step",SYMBOL_VOLUME_STEP);
   CaptureIntegerProperty("symbol_swap_mode",SYMBOL_SWAP_MODE);
   CaptureDoubleProperty("symbol_swap_long",SYMBOL_SWAP_LONG);
   CaptureDoubleProperty("symbol_swap_short",SYMBOL_SWAP_SHORT);
   CaptureIntegerProperty("symbol_swap_rollover3days",SYMBOL_SWAP_ROLLOVER3DAYS);
   CaptureIntegerProperty("symbol_trade_calc_mode",SYMBOL_TRADE_CALC_MODE);
   if(!SameContext())
     {
      ArrayFree(g_spec_keys);
      ArrayFree(g_spec_values);
      g_context_valid=false;
      g_spec_status="UNSTABLE_NOT_CAPTURED";
      return;
     }
   if(!g_spec_buffer_ok)
     {
      g_spec_status="NOT_CAPTURED_BUFFER_FAILURE";
      return;
     }
   g_spec_capture_time=capture_time;
   g_spec_status="CURRENT_ONLY_NOT_HISTORICAL";
  }

void PutCurrentSpec(const int handle)
  {
   // Only publish the already guarded snapshot; no terminal properties are read here.
   Put(handle,"spec_snapshot_status",g_spec_status);
   Put(handle,"spec_snapshot_historical","false");
   Put(handle,"spec_snapshot_capture_time_gmt",g_spec_capture_time);
   for(int i=0;i<ArraySize(g_spec_keys);i++) Put(handle,g_spec_keys[i],g_spec_values[i]);
  }

// A non-shared read/write handle locks cooperating exports without truncation.
int AcquirePrefixLock(const string filename)
  {
   if(FileIsExist(filename,FILE_COMMON)) return INVALID_HANDLE;
   ResetLastError();
   int handle=FileOpen(filename,FILE_READ|FILE_WRITE|FILE_BIN|FILE_COMMON);
   if(handle==INVALID_HANDLE) return INVALID_HANDLE;
   if(FileSize(handle)!=0)
     {
      FileClose(handle);
      return INVALID_HANDLE;
     }
   FileWriteString(handle,"read_only_quote_export_lock\r\n");
   FileFlush(handle);
   return handle;
  }

int OpenStaging(const string filename)
  {
   if(FileIsExist(filename,FILE_COMMON)) return INVALID_HANDLE;
   return FileOpen(filename,FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON,'\t',CP_UTF8);
  }

bool FlushAndClose(const int handle)
  {
   ResetLastError();
   FileFlush(handle);
   int error=GetLastError();
   if(error!=0)
     {
      g_write_ok=false;
      g_write_error=error;
     }
   ResetLastError();
   FileClose(handle);
   error=GetLastError();
   if(error!=0)
     {
      g_write_ok=false;
      g_write_error=error;
     }
   return g_write_ok;
  }

void OnStart()
  {
   if(!SafePrefix(InpPrefix))
     {
      Print("QUOTE_EXPORT_FAILED: unsafe output prefix; no file or quote request.");
      return;
     }
   string raw_file=InpPrefix+"_ticks.tsv";
   string receipt_file=InpPrefix+"_receipt.tsv";
   string raw_part=InpPrefix+"_ticks.part.tsv";
   string receipt_part=InpPrefix+"_receipt.part.tsv";
   string lock_file=InpPrefix+".lock";
   if(FileIsExist(raw_file,FILE_COMMON) || FileIsExist(receipt_file,FILE_COMMON) ||
      FileIsExist(raw_part,FILE_COMMON) || FileIsExist(receipt_part,FILE_COMMON))
     {
      Print("QUOTE_EXPORT_FAILED: output or partial file already exists; use a new prefix.");
      return;
     }
   int lock=AcquirePrefixLock(lock_file);
   if(lock==INVALID_HANDLE)
     {
      Print("QUOTE_EXPORT_FAILED: prefix lock exists or cannot be acquired.");
      return;
     }

   MqlTick saved[];
   ulong from_msc=0,end_msc=0;
   bool valid_input=(StringLen(InpSymbol)>0 && InpFromDate>0 && InpEndDate>InpFromDate &&
                     (InpEndDate-InpFromDate)<=86400 &&
                     InpConnectionWaitSeconds>=0 && InpConnectionWaitSeconds<=30);
   if(!valid_input) g_reason="invalid_symbol_dates_window_over_24h_or_connection_wait";
   else
     {
      from_msc=(ulong)InpFromDate*1000;
      end_msc=(ulong)InpEndDate*1000;
      if(WaitForExistingConnection() && ReadContext(g_server,g_company,g_account_mode))
        {
         g_context_valid=true;
         long symbol_exists=0;
         ResetLastError();
         bool symbol_available=SymbolInfoInteger(InpSymbol,SYMBOL_EXIST,symbol_exists);
         int symbol_error=GetLastError();
         long custom_symbol=0;
         ResetLastError();
         bool custom_available=SymbolInfoInteger(InpSymbol,SYMBOL_CUSTOM,custom_symbol);
         int custom_error=GetLastError();
         g_symbol_custom_observed=custom_available ? BoolText(custom_symbol!=0) : "NOT_AVAILABLE";
         if(!symbol_available || symbol_exists!=1)
           {
            g_status="FAILED_SYMBOL";
            g_reason="requested_symbol_not_available_without_changing_market_watch";
            g_guard_error=symbol_error;
           }
         else if(!custom_available || custom_error!=0 || custom_symbol!=0)
           {
            g_status="FAILED_SYMBOL";
            g_reason="custom_symbol_or_custom_status_unavailable";
            g_guard_error=custom_error;
           }
         else
           {
            g_symbol_eligible=true;
            g_status="PARTIAL";
            g_reason="request_not_completed_without_error";
            for(int attempt=0;attempt<3;attempt++)
              {
               if(IsStopped()) { g_reason="script_stopped"; break; }
               if(!SameContext()) { g_context_valid=false; break; }
               MqlTick received[];
               ResetLastError();
               ulong started=GetTickCount64();
               int copied=CopyTicksRange(InpSymbol,received,COPY_TICKS_ALL,from_msc,end_msc-1);
               int copy_error=GetLastError();
               ulong elapsed=GetTickCount64()-started;
               int available=ArraySize(received);
               g_attempts=attempt+1;
               g_last_error=copy_error;
               g_copy_returned_count=copied;
               g_attempt_errors[attempt]=copy_error;
               g_attempt_counts[attempt]=copied;
               g_attempt_array_rows[attempt]=available;
               g_attempt_elapsed_ms[attempt]=elapsed;
               bool clean=(copied>0 && copy_error==0 && copied==available);
               if(available>ArraySize(saved) || clean)
                 {
                  ArrayResize(saved,available);
                  if(ArrayCopy(saved,received,0,0,available)!=available)
                    {
                     g_status="PARTIAL";
                     g_reason="unable_to_retain_returned_tick_array";
                     break;
                    }
                  g_selected_attempt=attempt+1;
                 }
               if(!SameContext()) { g_context_valid=false; break; }
               if(clean)
                 {
                  g_api_returned_without_error=true;
                  g_status="SAVED_NOT_PROVEN";
                  g_reason="";
                  break;
                 }
               if(attempt<2 && !IsStopped()) Sleep(1000);
              }
            if(g_status=="PARTIAL" && ArraySize(saved)==0)
              {
               g_status="EMPTY";
               g_reason="no_tick_rows_returned_in_three_attempts_or_before_stop";
              }
           }
        }
     }

   int count=ArraySize(saved);
   int out_of_window=0,backward=0,crossed=0,invalid_quotes=0;
   for(int i=0;i<count;i++)
     {
      if(saved[i].time_msc<(long)from_msc || saved[i].time_msc>=(long)end_msc) out_of_window++;
      if(i>0 && saved[i].time_msc<saved[i-1].time_msc) backward++;
      if(MathIsValidNumber(saved[i].bid) && MathIsValidNumber(saved[i].ask) && saved[i].ask<saved[i].bid) crossed++;
      if(!MathIsValidNumber(saved[i].bid) || !MathIsValidNumber(saved[i].ask) ||
         !MathIsValidNumber(saved[i].last) || !MathIsValidNumber(saved[i].volume_real) ||
         saved[i].bid<=0 || saved[i].ask<=0 || saved[i].last<0 || saved[i].volume_real<0) invalid_quotes++;
     }
   if(g_status=="SAVED_NOT_PROVEN" && (out_of_window>0 || backward>0 || crossed>0 || invalid_quotes>0))
     {
      g_status="PARTIAL";
      g_reason="raw_rows_include_bounds_order_or_quote_validation_failures";
     }

   CaptureCurrentSpec();

   string published_raw="";
   int raw_rows_written=0;
   if(count>0)
     {
      g_write_ok=true;
      int raw=OpenStaging(raw_part);
      if(raw==INVALID_HANDLE)
        {
         g_status="FAILED_FILE_WRITE";
         g_reason="unable_to_create_raw_staging_file";
         g_write_error=GetLastError();
        }
      else
        {
         ResetLastError();
         RecordWrite(FileWrite(raw,"time","time_msc","bid","ask","last","volume","flags","volume_real"));
         for(int i=0;i<count && g_write_ok;i++)
           {
            ResetLastError();
            RecordWrite(FileWrite(raw,IntText((long)saved[i].time),IntText(saved[i].time_msc),
                                  RealText(saved[i].bid),RealText(saved[i].ask),RealText(saved[i].last),
                                  ULongText(saved[i].volume),IntText((long)saved[i].flags),RealText(saved[i].volume_real)));
            if(g_write_ok) raw_rows_written++;
           }
         if(FlushAndClose(raw) && raw_rows_written==count)
           {
            ResetLastError();
            if(FileMove(raw_part,FILE_COMMON,raw_file,FILE_COMMON)) published_raw=raw_file;
            else
              {
               g_status="FAILED_NO_OVERWRITE";
               g_reason="raw_destination_exists_or_publication_failed";
               g_write_error=GetLastError();
              }
           }
         else
           {
            g_status="FAILED_FILE_WRITE";
            g_reason="raw_staging_write_incomplete_partial_file_retained";
           }
        }
     }

   // Receipt is published last. A clean API return never proves tick completeness.
   g_write_ok=true;
   int receipt=OpenStaging(receipt_part);
   if(receipt==INVALID_HANDLE) Print("QUOTE_EXPORT_FAILED: cannot create receipt staging file.");
   else
     {
      ResetLastError();
      RecordWrite(FileWrite(receipt,"field","value"));
      Put(receipt,"schema_version","1");
      Put(receipt,"export_status",g_status);
      Put(receipt,"failure_reason",g_reason);
      Put(receipt,"symbol",InpSymbol);
      Put(receipt,"server",g_server);
      Put(receipt,"company",g_company);
      Put(receipt,"account_mode",g_account_mode);
      Put(receipt,"symbol_custom_observed",g_symbol_custom_observed);
      Put(receipt,"capture_time_gmt",GmtText(TimeGMT()));
      Put(receipt,"request_from_gmt",GmtText(InpFromDate));
      Put(receipt,"request_end_exclusive_gmt",GmtText(InpEndDate));
      Put(receipt,"request_from_msc",ULongText(from_msc));
      Put(receipt,"request_end_exclusive_msc",ULongText(end_msc));
      Put(receipt,"copy_to_inclusive_msc",end_msc>0 ? ULongText(end_msc-1) : "NOT_REQUESTED");
      Put(receipt,"window_end_exclusive","true");
      Put(receipt,"max_window_seconds","86400");
      Put(receipt,"copy_mode","COPY_TICKS_ALL");
      Put(receipt,"raw_file",published_raw);
      Put(receipt,"raw_partial_file",published_raw=="" && raw_rows_written>0 ? raw_part : "");
      Put(receipt,"row_count",IntText(count));
      Put(receipt,"raw_rows_written",IntText(raw_rows_written));
      Put(receipt,"first_time_msc",count>0 ? IntText(saved[0].time_msc) : "NONE");
      Put(receipt,"last_time_msc",count>0 ? IntText(saved[count-1].time_msc) : "NONE");
      Put(receipt,"last_error",IntText(g_last_error));
      Put(receipt,"guard_error",IntText(g_guard_error));
      Put(receipt,"write_error",IntText(g_write_error));
      Put(receipt,"attempts",IntText(g_attempts));
      Put(receipt,"retries",IntText(g_attempts>0 ? g_attempts-1 : 0));
      Put(receipt,"selected_attempt",IntText(g_selected_attempt));
      Put(receipt,"raw_retention_scope","SELECTED_ATTEMPT_ONLY");
      Put(receipt,"all_attempt_raw_arrays_retained","false");
      Put(receipt,"copy_returned_count",IntText(g_copy_returned_count));
      Put(receipt,"api_returned_without_error",BoolText(g_api_returned_without_error));
      Put(receipt,"connection_wait_limit_seconds",IntText(InpConnectionWaitSeconds));
      Put(receipt,"connection_wait_elapsed_ms",ULongText(g_connection_wait_ms));
      Put(receipt,"terminal_trade_allowed_observed",BoolText(g_terminal_trade_allowed));
      Put(receipt,"mql_trade_allowed_observed",BoolText(g_mql_trade_allowed));
      Put(receipt,"out_of_window_rows",IntText(out_of_window));
      Put(receipt,"backward_time_rows",IntText(backward));
      Put(receipt,"crossed_quote_rows",IntText(crossed));
      Put(receipt,"invalid_quote_rows",IntText(invalid_quotes));
      Put(receipt,"same_millisecond_rows_preserved","true");
      Put(receipt,"row_order_preserved","true");
      Put(receipt,"deduplication_applied","false");
      Put(receipt,"synchronization_status","NOT_PROVEN");
      Put(receipt,"historical_tick_completeness","NOT_PROVEN");
      Put(receipt,"live_trading","false");
      Put(receipt,"ready_for_live","false");
      Put(receipt,"live_allowed","false");
      Put(receipt,"opened_trades","false");
      Put(receipt,"safety_upper_live_trading","false");
      Put(receipt,"safety_upper_ready_for_live","false");
      Put(receipt,"safety_upper_live_allowed","false");
      Put(receipt,"safety_upper_opened_trades","false");
      Put(receipt,"costs_verified","false");
      Put(receipt,"execution_verified","false");
      Put(receipt,"historical_costs_verified","false");
      Put(receipt,"cfd_fills_verified","false");
      Put(receipt,"cash_contract_mapping_verified","false");
      Put(receipt,"quote_export_is_fill_evidence","false");
      Put(receipt,"account_identity_captured","false");
      Put(receipt,"limitation","Raw terminal quotes and current symbol specifications do not establish historical costs, executable fills, profit, or full historical synchronization.");
      for(int attempt=0;attempt<g_attempts;attempt++)
        {
         string key="attempt_"+IntText(attempt+1);
         Put(receipt,key+"_returned_count",IntText(g_attempt_counts[attempt]));
         Put(receipt,key+"_array_rows",IntText(g_attempt_array_rows[attempt]));
         Put(receipt,key+"_last_error",IntText(g_attempt_errors[attempt]));
         Put(receipt,key+"_elapsed_ms",ULongText(g_attempt_elapsed_ms[attempt]));
        }
      PutCurrentSpec(receipt);
      if(FlushAndClose(receipt))
        {
         ResetLastError();
         if(FileMove(receipt_part,FILE_COMMON,receipt_file,FILE_COMMON))
            PrintFormat("QUOTE_EXPORT %s: %d raw rows; receipt=%s",g_status,count,receipt_file);
         else PrintFormat("QUOTE_EXPORT_FAILED: receipt destination exists or publication failed; error=%d",GetLastError());
        }
      else Print("QUOTE_EXPORT_FAILED: receipt staging write incomplete; partial file retained.");
     }
   FileClose(lock);
   FileDelete(lock_file,FILE_COMMON);
  }
