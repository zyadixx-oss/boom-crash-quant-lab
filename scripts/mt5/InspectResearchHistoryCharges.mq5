#property strict
#property version "1.00"
#property description "Private read-only allowed-symbol history charges; no prices, PnL or record identifiers."
// These input datetime values have server_time semantics, with no UTC conversion.
input datetime InpFromDate = D'2025.10.01 00:00:00';
input datetime InpEndDate = 0; // Resolve once from the existing server's TimeCurrent.
input string   InpPrefix = "ResearchHistoryCharges_20251001";
input int      InpConnectionWaitSeconds = 30;
string g_server="",g_company="",g_mode="UNKNOWN";
string g_status="FAILED_INPUT",g_reason="";
bool g_terminal_allowed=false,g_program_allowed=false;
int g_guard_error=0,g_write_error=0;
bool g_write_ok=true;
ulong g_wait_ms=0;
string g_currency="",g_currency_capture="NOT_CAPTURED";
int g_currency_error=0;
bool g_context_failed=false;
string IntText(const long value) { return IntegerToString(value); }
string BoolText(const bool value) { return value ? "true" : "false"; }
string CleanValue(string value)
  {
   StringReplace(value,"\t"," ");
   StringReplace(value,"\r"," ");
   StringReplace(value,"\n"," ");
   return value;
  }
bool SafePrefix(const string value)
  {
   int size=StringLen(value);
   if(size<1 || size>64) return false;
   for(int i=0;i<size;i++)
     {
      ushort c=StringGetCharacter(value,i);
      bool letter=(c>='a' && c<='z') || (c>='A' && c<='Z');
      if(!letter && !(c>='0' && c<='9') && c!='_' && c!='-') return false;
     }
   string lower=value;
   StringToLower(lower);
   if(lower=="con" || lower=="prn" || lower=="aux" || lower=="nul") return false;
   if(size==4 && (StringSubstr(lower,0,3)=="com" || StringSubstr(lower,0,3)=="lpt"))
     {
      ushort last=StringGetCharacter(lower,3);
      if(last>='1' && last<='9') return false;
     }
   return true;
  }
bool TradingDisabled()
  {
   ResetLastError();
   long terminal=TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   int terminal_error=GetLastError();
   ResetLastError();
   long program=MQLInfoInteger(MQL_TRADE_ALLOWED);
   int program_error=GetLastError();
   g_terminal_allowed=(terminal!=0);
   g_program_allowed=(program!=0);
   g_guard_error=(terminal_error!=0 ? terminal_error : program_error);
   return terminal_error==0 && program_error==0 && terminal==0 && program==0;
  }
bool Connected()
  {
   ResetLastError();
   long connected=TerminalInfoInteger(TERMINAL_CONNECTED);
   int error=GetLastError();
   if(error!=0) g_guard_error=error;
   return error==0 && connected==1;
  }
bool WaitForConnection()
  {
   ulong started=GetTickCount64();
   ulong limit=(ulong)InpConnectionWaitSeconds*1000;
   while(true)
     {
      if(!TradingDisabled())
        {
         g_status="FAILED_UNSAFE";
         g_reason="terminal_or_script_trading_not_disabled_or_guard_unavailable";
         g_wait_ms=GetTickCount64()-started;
         return false;
        }
      if(Connected()) { g_wait_ms=GetTickCount64()-started; return true; }
      g_wait_ms=GetTickCount64()-started;
      if(IsStopped() || g_wait_ms>=limit) break;
      int remaining=(int)(limit-g_wait_ms);
      Sleep(remaining<100 ? remaining : 100);
     }
   g_status="FAILED_DISCONNECTED";
   g_reason="existing_server_connection_unavailable_after_bounded_wait";
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
   if(server_error!=0 || company_error!=0 || mode_error!=0 ||
      StringFind(lower,"deriv")!=0 || StringLen(company)==0)
     {
      g_status="FAILED_ACCOUNT_CONTEXT";
      g_reason="unavailable_or_non_deriv_server_context";
      g_guard_error=(server_error!=0 ? server_error : (company_error!=0 ? company_error : mode_error));
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
   if(server!=g_server || company!=g_company || mode!=g_mode)
     {
      g_status="FAILED_ACCOUNT_CONTEXT";
      g_reason="server_company_or_account_mode_changed_during_query";
      return false;
     }
   return true;
  }
void RecordWrite(const uint bytes)
  {
   if(bytes==0)
     {
      g_write_ok=false;
      if(g_write_error==0) g_write_error=GetLastError();
     }
  }
void Put(const int handle,const string key,const string value)
  {
   ResetLastError();
   RecordWrite(FileWrite(handle,key,CleanValue(value)));
  }
int AcquireLock(const string filename)
  {
   if(FileIsExist(filename,FILE_COMMON)) return INVALID_HANDLE;
   int handle=FileOpen(filename,FILE_READ|FILE_WRITE|FILE_BIN|FILE_COMMON);
   if(handle==INVALID_HANDLE) return INVALID_HANDLE;
   if(FileSize(handle)!=0) { FileClose(handle); return INVALID_HANDLE; }
   FileWriteString(handle,"read_only_history_availability_lock\r\n");
   FileFlush(handle);
   return handle;
  }

string RealText(const double value) { return StringFormat("%.17g",value); }
bool SafeCurrency(const string value)
  {
   int n=StringLen(value);
   if(n<3 || n>12) return false;
   for(int i=0;i<n;i++)
     {
      ushort c=StringGetCharacter(value,i);
      if(!(c>='A' && c<='Z') && !(c>='a' && c<='z') && !(c>='0' && c<='9')) return false;
     }
   return true;
  }
bool ChargeContext()
  {
   if(g_context_failed) return false;
   if(!SameContext()) { g_context_failed=true; return false; }
   ResetLastError();
   string currency=AccountInfoString(ACCOUNT_CURRENCY);
   int error=GetLastError();
   if(error!=0 || !SafeCurrency(currency) || (g_currency!="" && currency!=g_currency))
     {
      g_context_failed=true; g_currency_error=error;
      g_status="FAILED_ACCOUNT_CONTEXT"; g_reason="currency_unavailable_unsafe_or_changed";
      return false;
     }
   if(g_currency=="")
     {
      g_currency=currency;
      g_currency_capture=TimeToString(TimeGMT(),TIME_DATE|TIME_SECONDS)+" GMT";
     }
   if(!SameContext()) { g_context_failed=true; return false; }
   return true;
  }
bool AllowedSymbol(const string symbol)
  {
   return symbol=="Boom 300 Index" || symbol=="Boom 500 Index" || symbol=="Boom 600 Index" ||
          symbol=="Boom 900 Index" || symbol=="Boom 1000 Index" || symbol=="Crash 300 Index" ||
          symbol=="Crash 500 Index" || symbol=="Crash 600 Index" || symbol=="Crash 900 Index" || symbol=="Crash 1000 Index";
  }
string ReadCharge(const ulong ticket,const ENUM_DEAL_PROPERTY_DOUBLE property,int &error,bool &available,double &value)
  {
   ResetLastError();
   available=HistoryDealGetDouble(ticket,property,value);
   error=GetLastError();
   return available && error==0 ? RealText(value) : "NOT_AVAILABLE";
  }
bool CloseStaging(const int handle)
  {
   ResetLastError(); FileFlush(handle); int error=GetLastError();
   if(error!=0) { g_write_ok=false; g_write_error=error; }
   ResetLastError(); FileClose(handle); error=GetLastError();
   if(error!=0) { g_write_ok=false; g_write_error=error; }
   return g_write_ok;
  }
int OpenPrivateStaging(const string filename)
  {
   if(FileIsExist(filename,FILE_COMMON)) return INVALID_HANDLE;
   ResetLastError();
   return FileOpen(filename,FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON,'\t',CP_UTF8);
  }
void OnStart()
  {
   if(!SafePrefix(InpPrefix)) { Print("PRIVATE_CHARGES_FAILED: unsafe prefix."); return; }
   string raw=InpPrefix+"_charges.tsv",raw_part=InpPrefix+"_charges.part.tsv";
   string receipt=InpPrefix+"_receipt.tsv",receipt_part=InpPrefix+"_receipt.part.tsv";
   string lock_file=InpPrefix+"_charges.lock";
   if(FileIsExist(raw,FILE_COMMON) || FileIsExist(raw_part,FILE_COMMON) ||
      FileIsExist(receipt,FILE_COMMON) || FileIsExist(receipt_part,FILE_COMMON))
     { Print("PRIVATE_CHARGES_FAILED: output exists; use a new prefix."); return; }
   int lock=AcquireLock(lock_file);
   if(lock==INVALID_HANDLE) { Print("PRIVATE_CHARGES_FAILED: lock unavailable."); return; }
   datetime server_clock=0,end_date=0;
   bool select_called=false,selected=false,deals_called=false;
   bool before_query=false,after_query=false,before_loop=false,after_loop=false,loop_complete=false,have_time=false;
   int clock_error=0,select_error=0,deals_error=0,total=0,processed=0,matched=0,unrelated=0,nontrade=0;
   int unclassified=0,property_errors=0,numeric_invalid=0,outside=0,backwards=0,written=0;
   int first_property_error=0,last_property_error=0;
   long first_time=0,last_time=0,minimum_time=0,maximum_time=0;
   ulong query_ms=0,loop_ms=0;
   string published_raw="";
   int raw_handle=INVALID_HANDLE;
   if(InpFromDate<=0 || InpEndDate<0 || InpConnectionWaitSeconds<0 || InpConnectionWaitSeconds>30)
      g_reason="invalid_start_end_or_connection_wait";
   else if(WaitForConnection() && ReadContext(g_server,g_company,g_mode) && ChargeContext())
     {
      ResetLastError(); server_clock=TimeCurrent(); clock_error=GetLastError();
      end_date=(InpEndDate==0 ? server_clock : InpEndDate);
      if(clock_error!=0 || server_clock<=0 || end_date<=InpFromDate || end_date>server_clock ||
         (end_date-InpFromDate)>400*86400)
        { g_status="FAILED_INPUT"; g_reason="invalid_or_future_server_window_or_span_over_400_days"; }
      else if(IsStopped()) { g_status="FAILED_STOPPED"; g_reason="script_stopped_before_query"; }
      else if((before_query=ChargeContext()))
        {
         ulong started=GetTickCount64();
         ResetLastError(); select_called=true; selected=HistorySelect(InpFromDate,end_date); select_error=GetLastError();
         if(selected && select_error==0)
           { ResetLastError(); total=HistoryDealsTotal(); deals_error=GetLastError(); deals_called=true; }
         query_ms=GetTickCount64()-started;
         after_query=ChargeContext();
         if(after_query)
           {
            if(!selected || select_error!=0 || !deals_called || deals_error!=0 || total<0)
              { g_status="FAILED_SELECTION_OR_COUNT"; g_reason="history_selection_or_count_unavailable"; }
            else if(total>50000) { g_status="FAILED_RECORD_LIMIT"; g_reason="selected_list_exceeds_50000_no_records_read"; }
            else if((before_loop=ChargeContext()))
              {
               raw_handle=OpenPrivateStaging(raw_part);
               if(raw_handle==INVALID_HANDLE)
                 { g_status="FAILED_FILE_WRITE"; g_reason="cannot_create_private_raw_staging"; g_write_error=GetLastError(); }
               else
                 {
                  ResetLastError();
                  RecordWrite(FileWrite(raw_handle,"time_msc","symbol","deal_type","deal_entry","volume","commission","swap","fee",
                      "time_error_code","entry_error_code","volume_error_code","commission_error_code","swap_error_code","fee_error_code",
                      "numeric_valid","time_in_requested_server_bounds","row_context_stable"));
                  ulong loop_started=GetTickCount64();
                  for(int i=0;i<total && g_write_ok;i++)
                    {
                     if(IsStopped()) { g_status="FAILED_STOPPED"; g_reason="script_stopped_during_capture"; break; }
                     if(!ChargeContext()) break;
                     ResetLastError(); ulong ticket=HistoryDealGetTicket(i); int ticket_error=GetLastError(); processed++;
                     string symbol=""; long type=-1;
                     ResetLastError(); bool symbol_ok=(ticket!=0 && ticket_error==0) && HistoryDealGetString(ticket,DEAL_SYMBOL,symbol);
                     int symbol_error=GetLastError();
                     ResetLastError(); bool type_ok=(ticket!=0 && ticket_error==0) && HistoryDealGetInteger(ticket,DEAL_TYPE,type);
                     int type_error=GetLastError();
                     if(ticket==0 || ticket_error!=0 || !symbol_ok || symbol_error!=0 || !type_ok || type_error!=0)
                       {
                        unclassified++; property_errors++;
                        int e=(ticket_error!=0 ? ticket_error : (symbol_error!=0 ? symbol_error : type_error));
                        if(first_property_error==0 && e!=0) first_property_error=e; last_property_error=e;
                        continue;
                       }
                     if(type!=DEAL_TYPE_BUY && type!=DEAL_TYPE_SELL) { nontrade++; continue; }
                     if(!AllowedSymbol(symbol)) { unrelated++; continue; }
                     matched++;
                     long time_msc=0,entry=-1;
                     ResetLastError(); bool time_ok=HistoryDealGetInteger(ticket,DEAL_TIME_MSC,time_msc); int time_error=GetLastError();
                     ResetLastError(); bool entry_ok=HistoryDealGetInteger(ticket,DEAL_ENTRY,entry); int entry_error=GetLastError();
                     double volume=0,commission=0,swap=0,fee=0;
                     int volume_error=0,commission_error=0,swap_error=0,fee_error=0;
                     bool volume_ok=false,commission_ok=false,swap_ok=false,fee_ok=false;
                     string volume_text=ReadCharge(ticket,DEAL_VOLUME,volume_error,volume_ok,volume);
                     string commission_text=ReadCharge(ticket,DEAL_COMMISSION,commission_error,commission_ok,commission);
                     string swap_text=ReadCharge(ticket,DEAL_SWAP,swap_error,swap_ok,swap);
                     string fee_text=ReadCharge(ticket,DEAL_FEE,fee_error,fee_ok,fee);
                     bool properties_ok=time_ok && time_error==0 && entry_ok && entry_error==0 &&
                          volume_ok && volume_error==0 && commission_ok && commission_error==0 &&
                          swap_ok && swap_error==0 && fee_ok && fee_error==0;
                     if(!properties_ok)
                       {
                        property_errors++;
                        int errors[6]; errors[0]=time_error; errors[1]=entry_error; errors[2]=volume_error;
                        errors[3]=commission_error; errors[4]=swap_error; errors[5]=fee_error;
                        for(int j=0;j<6;j++) if(errors[j]!=0)
                          { if(first_property_error==0) first_property_error=errors[j]; last_property_error=errors[j]; }
                       }
                     bool numeric_ok=properties_ok && MathIsValidNumber(volume) && volume>0 &&
                          MathIsValidNumber(commission) && MathIsValidNumber(swap) && MathIsValidNumber(fee) &&
                          entry>=DEAL_ENTRY_IN && entry<=DEAL_ENTRY_OUT_BY;
                     bool in_bounds=time_ok && time_error==0 && time_msc>=(long)InpFromDate*1000 && time_msc<=(long)end_date*1000+999;
                     if(!numeric_ok) numeric_invalid++;
                     if(!in_bounds) outside++;
                     if(time_ok && time_error==0)
                       {
                        if(!have_time) { first_time=time_msc; minimum_time=time_msc; maximum_time=time_msc; have_time=true; }
                        else { if(time_msc<last_time) backwards++; if(time_msc<minimum_time) minimum_time=time_msc; if(time_msc>maximum_time) maximum_time=time_msc; }
                        last_time=time_msc;
                       }
                     bool row_context=ChargeContext();
                     ResetLastError();
                     RecordWrite(FileWrite(raw_handle,time_ok && time_error==0 ? IntText(time_msc) : "NOT_AVAILABLE",symbol,IntText(type),
                          entry_ok && entry_error==0 ? IntText(entry) : "NOT_AVAILABLE",volume_text,commission_text,swap_text,fee_text,
                          IntText(time_error),IntText(entry_error),IntText(volume_error),IntText(commission_error),IntText(swap_error),IntText(fee_error),
                          BoolText(numeric_ok),BoolText(in_bounds),BoolText(row_context)));
                     if(g_write_ok) written++;
                     if(!row_context) break;
                    }
                  loop_ms=GetTickCount64()-loop_started;
                  after_loop=ChargeContext();
                  loop_complete=processed==total && after_loop && g_write_ok && !IsStopped();
                  if(IsStopped() && after_loop) { g_status="FAILED_STOPPED"; g_reason="script_stopped_at_capture_completion"; }
                  if(after_loop && g_write_ok && g_status!="FAILED_STOPPED")
                    {
                     if(property_errors>0 || numeric_invalid>0 || outside>0)
                       { g_status="FAILED_PROPERTY_OR_NUMERIC"; g_reason="retained_diagnostics_include_unavailable_invalid_or_outside_rows"; }
                     else if(loop_complete)
                       { g_status=(matched>0 ? "CHARGE_COMPONENTS_OBSERVED_NOT_PROVEN" : "NO_MATCHING_CHARGES_OBSERVED_NOT_PROVEN"); g_reason=""; }
                    }
                  if(!CloseStaging(raw_handle)) { g_status="FAILED_FILE_WRITE"; g_reason="raw_staging_incomplete_retained"; }
                  else
                    {
                     ResetLastError();
                     if(FileMove(raw_part,FILE_COMMON,raw,FILE_COMMON)) published_raw=raw;
                     else { g_status="FAILED_FILE_WRITE"; g_reason="raw_publication_failed_partial_retained"; g_write_error=GetLastError(); }
                    }
                 }
              }
           }
        }
     }
   // Only private files receive account-specific observations. Logs carry no counts, currency or filenames.
   g_write_ok=true;
   int file=OpenPrivateStaging(receipt_part);
   if(file==INVALID_HANDLE) Print("PRIVATE_CHARGES_FAILED: receipt staging unavailable.");
   else
     {
      ResetLastError(); RecordWrite(FileWrite(file,"field","value"));
      Put(file,"schema_version","1"); Put(file,"scope","PRIVATE_ALLOWED_SYMBOL_HISTORY_CHARGE_COMPONENTS_ONLY");
      Put(file,"status",g_status); Put(file,"failure_reason",g_reason);
      Put(file,"server",g_server); Put(file,"company",g_company); Put(file,"account_mode",g_mode);
      Put(file,"capture_time_gmt",TimeToString(TimeGMT(),TIME_DATE|TIME_SECONDS)+" GMT");
      Put(file,"account_currency_current",g_currency); Put(file,"account_currency_scope","CURRENT_NOT_HISTORICAL");
      Put(file,"account_currency_capture_time_gmt",g_currency_capture); Put(file,"account_currency_error",IntText(g_currency_error));
      Put(file,"historical_currency_applicability_verified","false");
      Put(file,"requested_time_semantics","server_time"); Put(file,"requested_bounds_are_utc","false");
      Put(file,"server_timezone_offset_verified","false"); Put(file,"input_start_server_epoch",IntText((long)InpFromDate));
      Put(file,"input_end_server_epoch",IntText((long)InpEndDate)); Put(file,"requested_start_server_epoch",IntText((long)InpFromDate));
      Put(file,"requested_end_server_epoch",end_date>0 ? IntText((long)end_date) : "NOT_RESOLVED");
      Put(file,"server_clock_at_query_epoch",server_clock>0 ? IntText((long)server_clock) : "NOT_READ");
      Put(file,"server_clock_error",IntText(clock_error)); Put(file,"server_clock_semantics","TimeCurrent_last_known_existing_server_time");
      Put(file,"end_resolved_from_timecurrent",BoolText(InpEndDate==0 && server_clock>0)); Put(file,"max_span_days","400");
      Put(file,"selection_bounds_semantics","HistorySelect_server_time_bounds_no_UTC_conversion");
      Put(file,"matched_time_bounds_semantics","inclusive_server_seconds_milliseconds_end_plus999");
      Put(file,"history_select_called",BoolText(select_called)); Put(file,"history_select_call_count",select_called ? "1" : "0");
      Put(file,"history_select_returned",BoolText(selected)); Put(file,"history_select_error",IntText(select_error));
      Put(file,"history_deals_total_called",BoolText(deals_called)); Put(file,"history_deals_total_returned",deals_called ? IntText(total) : "NOT_QUERIED");
      Put(file,"history_deals_total_error",IntText(deals_error)); Put(file,"maximum_selected_records","50000");
      Put(file,"processed_records",IntText(processed)); Put(file,"matched_records",IntText(matched)); Put(file,"raw_rows_written",IntText(written));
      Put(file,"unrelated_trade_records",IntText(unrelated)); Put(file,"nontrade_records",IntText(nontrade)); Put(file,"unclassified_records",IntText(unclassified));
      Put(file,"property_error_rows",IntText(property_errors)); Put(file,"numeric_invalid_rows",IntText(numeric_invalid));
      Put(file,"out_of_requested_bounds_rows",IntText(outside)); Put(file,"backward_matched_time_rows",IntText(backwards));
      Put(file,"first_property_error",IntText(first_property_error)); Put(file,"last_property_error",IntText(last_property_error));
      Put(file,"first_matched_time_msc",have_time ? IntText(first_time) : "NONE"); Put(file,"last_matched_time_msc",have_time ? IntText(last_time) : "NONE");
      Put(file,"minimum_matched_time_msc",have_time ? IntText(minimum_time) : "NONE"); Put(file,"maximum_matched_time_msc",have_time ? IntText(maximum_time) : "NONE");
      Put(file,"context_stable_before_query",BoolText(before_query)); Put(file,"context_stable_after_query",BoolText(after_query));
      Put(file,"context_stable_before_loop",BoolText(before_loop)); Put(file,"context_stable_after_loop",BoolText(after_loop)); Put(file,"selected_list_loop_complete",BoolText(loop_complete));
      Put(file,"query_elapsed_ms",StringFormat("%I64u",query_ms)); Put(file,"loop_elapsed_ms",StringFormat("%I64u",loop_ms));
      Put(file,"connection_wait_limit_seconds",IntText(InpConnectionWaitSeconds)); Put(file,"connection_wait_elapsed_ms",StringFormat("%I64u",g_wait_ms));
      Put(file,"terminal_trade_allowed_observed",BoolText(g_terminal_allowed)); Put(file,"mql_trade_allowed_observed",BoolText(g_program_allowed));
      Put(file,"guard_error",IntText(g_guard_error)); Put(file,"write_error",IntText(g_write_error)); Put(file,"raw_file",published_raw);
      Put(file,"raw_partial_file",published_raw=="" && raw_handle!=INVALID_HANDLE ? raw_part : ""); Put(file,"private_output","true");
      Put(file,"allowed_symbols","Boom/Crash exact Index names for 300,500,600,900,1000"); Put(file,"allowed_deal_types","BUY_SELL_ONLY");
      Put(file,"record_order_preserved","true"); Put(file,"deduplication_applied","false"); Put(file,"raw_retention_scope","SINGLE_SELECTION_MATCHING_ROWS_ONLY");
      Put(file,"raw_attempts",select_called ? "1" : "0"); Put(file,"unclassified_record_details_retained","false");
      Put(file,"matching_rows_with_errors_retained",BoolText(raw_handle!=INVALID_HANDLE && written==matched));
      Put(file,"current_cost_completeness","NOT_PROVEN"); Put(file,"historical_cost_completeness","NOT_PROVEN"); Put(file,"history_selection_completeness","NOT_PROVEN");
      Put(file,"same_server_same_mode_account_switch_detection","false"); Put(file,"context_comparison_scope","server_company_account_mode_current_currency_only");
      Put(file,"all_period_fees_established","false"); Put(file,"actual_slippage_evidence","false"); Put(file,"historical_contract_mapping_verified","false");
      Put(file,"live_trading","false"); Put(file,"ready_for_live","false"); Put(file,"live_allowed","false"); Put(file,"opened_trades","false");
      Put(file,"safety_upper_live_trading","false"); Put(file,"safety_upper_ready_for_live","false"); Put(file,"safety_upper_live_allowed","false"); Put(file,"safety_upper_opened_trades","false");
      Put(file,"costs_verified","false"); Put(file,"historical_costs_verified","false"); Put(file,"execution_verified","false"); Put(file,"cfd_fills_verified","false");
      Put(file,"prices_exported","false"); Put(file,"pnl_exported","false"); Put(file,"tickets_exported","false"); Put(file,"position_order_ids_exported","false");
      Put(file,"direct_account_identity_fields_exported","false"); Put(file,"private_financial_records_exported",BoolText(written>0));
      Put(file,"account_login_name_balance_recorded","false"); Put(file,"credentials_recorded","false");
      Put(file,"login_or_account_creation_attempted","false"); Put(file,"orders_sent","0");
      Put(file,"limitation","Private matching BUY/SELL deal charge components only; no prices, PnL, identity, pairing, full-period costs, historical currency applicability, completeness, broker-fill or slippage validation.");
      if(CloseStaging(file))
        {
         ResetLastError();
         if(FileMove(receipt_part,FILE_COMMON,receipt,FILE_COMMON)) Print("PRIVATE_CHARGES_CAPTURE_FINISHED: inspect private retained evidence.");
         else Print("PRIVATE_CHARGES_FAILED: receipt publication unavailable; partial retained.");
        }
      else Print("PRIVATE_CHARGES_FAILED: incomplete private receipt retained.");
     }
   FileClose(lock); FileDelete(lock_file,FILE_COMMON);
  }
