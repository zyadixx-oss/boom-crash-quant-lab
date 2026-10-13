#property strict
#property version "1.00"
#property description "Read-only account-history availability counts; no individual records."

// These input datetime values have server_time semantics, with no UTC conversion.
input datetime InpFromDate = D'2025.10.01 00:00:00';
input datetime InpEndDate = 0; // Resolve once from the existing server's TimeCurrent.
input string   InpPrefix = "ResearchHistoryAvailability_20251001";
input int      InpConnectionWaitSeconds = 30;

string g_server="",g_company="",g_mode="UNKNOWN";
string g_status="FAILED_INPUT",g_reason="";
bool g_terminal_allowed=false,g_program_allowed=false;
int g_guard_error=0,g_write_error=0;
bool g_write_ok=true;
ulong g_wait_ms=0;

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

void OnStart()
  {
   if(!SafePrefix(InpPrefix))
     {
      Print("HISTORY_AVAILABILITY_FAILED: unsafe prefix; no history query or file write.");
      return;
     }
   string output=InpPrefix+"_availability.tsv";
   string partial=InpPrefix+"_availability.part.tsv";
   string lock_file=InpPrefix+"_availability.lock";
   if(FileIsExist(output,FILE_COMMON) || FileIsExist(partial,FILE_COMMON))
     {
      Print("HISTORY_AVAILABILITY_FAILED: output already exists; use a new prefix.");
      return;
     }
   int lock=AcquireLock(lock_file);
   if(lock==INVALID_HANDLE)
     {
      Print("HISTORY_AVAILABILITY_FAILED: prefix lock exists or is unavailable.");
      return;
     }

   datetime server_clock=0,end_date=0;
   bool before=false,after=false,select_called=false,select_returned=false;
   bool deals_called=false,orders_called=false,counts_accepted=false;
   int select_error=0,deals_error=0,orders_error=0,clock_error=0;
   int deals_count=0,orders_count=0;
   ulong query_ms=0;
   if(InpFromDate<=0 || InpEndDate<0 || InpConnectionWaitSeconds<0 || InpConnectionWaitSeconds>30)
      g_reason="invalid_start_end_or_connection_wait";
   else if(WaitForConnection() && ReadContext(g_server,g_company,g_mode))
     {
      ResetLastError();
      server_clock=TimeCurrent();
      clock_error=GetLastError();
      end_date=(InpEndDate==0 ? server_clock : InpEndDate);
      if(clock_error!=0 || server_clock<=0 || end_date<=InpFromDate ||
         end_date>server_clock || (end_date-InpFromDate)>400*86400)
        {
         g_status="FAILED_INPUT";
         g_reason="invalid_or_future_server_time_window_or_span_over_400_days";
        }
      else if(IsStopped())
        {
         g_status="FAILED_STOPPED";
         g_reason="script_stopped_before_history_query";
        }
      else if((before=SameContext()))
        {
         ulong started=GetTickCount64();
         ResetLastError();
         select_called=true;
         select_returned=HistorySelect(InpFromDate,end_date);
         select_error=GetLastError();
         if(select_returned && select_error==0)
           {
            ResetLastError();
            deals_count=HistoryDealsTotal();
            deals_error=GetLastError();
            deals_called=true;
            ResetLastError();
            orders_count=HistoryOrdersTotal();
            orders_error=GetLastError();
            orders_called=true;
           }
         query_ms=GetTickCount64()-started;
         after=SameContext();
         if(after)
           {
            counts_accepted=select_returned && select_error==0 && deals_called && orders_called &&
                            deals_error==0 && orders_error==0 && deals_count>=0 && orders_count>=0;
            g_status=(counts_accepted ? "COUNTS_OBSERVED_NOT_PROVEN" : "FAILED_SELECTION_OR_COUNTS");
            g_reason=(counts_accepted ? "" : "selection_return_error_or_nonnegative_counts_check_failed");
           }
        }
     }

   // Serialize only the completed count observation; no further account/history reads.
   ResetLastError();
   int file=FileOpen(partial,FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON,'\t',CP_UTF8);
   if(file==INVALID_HANDLE) PrintFormat("HISTORY_AVAILABILITY_FAILED: cannot open staging file; error=%d",GetLastError());
   else
     {
      ResetLastError();
      RecordWrite(FileWrite(file,"field","value"));
      Put(file,"schema_version","1");
      Put(file,"scope","ACCOUNT_HISTORY_AVAILABILITY_COUNTS_ONLY");
      Put(file,"status",g_status);
      Put(file,"failure_reason",g_reason);
      Put(file,"server",g_server);
      Put(file,"company",g_company);
      Put(file,"account_mode",g_mode);
      Put(file,"capture_time_gmt",TimeToString(TimeGMT(),TIME_DATE|TIME_SECONDS)+" GMT");
      Put(file,"requested_time_semantics","server_time");
      Put(file,"requested_bounds_are_utc","false");
      Put(file,"server_timezone_offset_verified","false");
      Put(file,"input_start_server_epoch",IntText((long)InpFromDate));
      Put(file,"input_end_server_epoch",IntText((long)InpEndDate));
      Put(file,"requested_start_server_epoch",IntText((long)InpFromDate));
      Put(file,"requested_end_server_epoch",end_date>0 ? IntText((long)end_date) : "NOT_RESOLVED");
      Put(file,"server_clock_at_query_epoch",server_clock>0 ? IntText((long)server_clock) : "NOT_READ");
      Put(file,"server_clock_error",IntText(clock_error));
      Put(file,"server_clock_semantics","TimeCurrent_last_known_existing_server_time");
      Put(file,"end_resolved_from_timecurrent",BoolText(InpEndDate==0 && server_clock>0));
      Put(file,"max_span_days","400");
      Put(file,"selection_bounds_semantics","HistorySelect_server_time_bounds_no_UTC_conversion");
      Put(file,"history_select_called",BoolText(select_called));
      Put(file,"history_select_call_count",select_called ? "1" : "0");
      Put(file,"history_select_returned",BoolText(select_returned));
      Put(file,"history_select_error",IntText(select_error));
      Put(file,"history_deals_total_called",BoolText(deals_called));
      Put(file,"history_deals_total_returned",deals_called ? IntText(deals_count) : "NOT_QUERIED");
      Put(file,"history_deals_total_error",IntText(deals_error));
      Put(file,"history_orders_total_called",BoolText(orders_called));
      Put(file,"history_orders_total_returned",orders_called ? IntText(orders_count) : "NOT_QUERIED");
      Put(file,"history_orders_total_error",IntText(orders_error));
      Put(file,"counts_accepted",BoolText(counts_accepted));
      Put(file,"context_stable_before_query",BoolText(before));
      Put(file,"context_stable_after_query",BoolText(after));
      Put(file,"query_elapsed_ms",StringFormat("%I64u",query_ms));
      Put(file,"connection_wait_limit_seconds",IntText(InpConnectionWaitSeconds));
      Put(file,"connection_wait_elapsed_ms",StringFormat("%I64u",g_wait_ms));
      Put(file,"terminal_trade_allowed_observed",BoolText(g_terminal_allowed));
      Put(file,"mql_trade_allowed_observed",BoolText(g_program_allowed));
      Put(file,"guard_error",IntText(g_guard_error));
      Put(file,"selection_scope","ALL_ACCOUNT_HISTORY_IN_REQUESTED_RANGE_NOT_SYMBOL_FILTERED");
      Put(file,"selection_completeness","NOT_PROVEN");
      Put(file,"whole_history_completeness","NOT_PROVEN");
      Put(file,"same_server_same_mode_account_switch_detection","false");
      Put(file,"context_comparison_scope","server_company_account_mode_only");
      Put(file,"boom_crash_relation_verified","false");
      Put(file,"deal_count_is_trade_fill_count","false");
      Put(file,"record_type_classification_performed","false");
      Put(file,"live_trading","false");
      Put(file,"ready_for_live","false");
      Put(file,"live_allowed","false");
      Put(file,"opened_trades","false");
      Put(file,"safety_upper_live_trading","false");
      Put(file,"safety_upper_ready_for_live","false");
      Put(file,"safety_upper_live_allowed","false");
      Put(file,"safety_upper_opened_trades","false");
      Put(file,"costs_verified","false");
      Put(file,"historical_costs_verified","false");
      Put(file,"execution_verified","false");
      Put(file,"cfd_fills_verified","false");
      Put(file,"prices_exported","false");
      Put(file,"pnl_exported","false");
      Put(file,"tickets_exported","false");
      Put(file,"individual_records_exported","false");
      Put(file,"personal_data_exported","false");
      Put(file,"account_login_name_balance_recorded","false");
      Put(file,"credentials_recorded","false");
      Put(file,"login_or_account_creation_attempted","false");
      Put(file,"orders_sent","0");
      Put(file,"output_file",output);
      Put(file,"limitation","Counts describe selected account-history lists only; no symbol relation, trade-fill classification, costs, prices, profit, or history completeness is established.");
      ResetLastError();
      FileFlush(file);
      int error=GetLastError();
      if(error!=0) { g_write_ok=false; g_write_error=error; }
      ResetLastError();
      FileClose(file);
      error=GetLastError();
      if(error!=0) { g_write_ok=false; g_write_error=error; }
      if(g_write_ok)
        {
         ResetLastError();
         if(FileMove(partial,FILE_COMMON,output,FILE_COMMON))
            PrintFormat("HISTORY_AVAILABILITY %s: receipt=%s",g_status,output);
         else PrintFormat("HISTORY_AVAILABILITY_FAILED: destination exists or publication failed; error=%d",GetLastError());
        }
      else PrintFormat("HISTORY_AVAILABILITY_FAILED: staging write incomplete; error=%d",g_write_error);
     }
   FileClose(lock);
   FileDelete(lock_file,FILE_COMMON);
  }
