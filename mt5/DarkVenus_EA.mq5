//+------------------------------------------------------------------+
//|                                              DarkVenus_EA.mq5    |
//|                                    Dark Venus Strategy for MT5   |
//|                          Bollinger Bands + Trailing Stop         |
//+------------------------------------------------------------------+
#property copyright "Dark Venus"
#property link      ""
#property version   "1.00"
#property strict

//--- Enum declarations (MUST be before inputs)
enum ENUM_DIRECTION 
{ 
    LONG_ONLY = 0,    // Long Only
    SHORT_ONLY = 1,   // Short Only
    BOTH = 2          // Both
};

enum ENUM_BB_STRATEGY 
{ 
    SELL_ABOVE_BUY_BELOW = 0,  // Sell Above Buy Below
    BUY_ABOVE_SELL_BELOW = 1   // Buy Above Sell Below
};

//--- Input Parameters
input string Sep1 = "═══ Direction de Trading ═══"; // ═══════════════
input ENUM_DIRECTION TradeDirection = BOTH; // Direction de trading

input string Sep2 = "═══ Bollinger Bands ═══"; // ═══════════════
input ENUM_BB_STRATEGY BBStrategy = SELL_ABOVE_BUY_BELOW; // Stratégie BB
input int BBPeriod = 20;           // Période BB
input double BBDeviation = 2.0;    // Déviation BB
input ENUM_APPLIED_PRICE BBSource = PRICE_CLOSE; // Source de prix

input string Sep3 = "═══ Trailing Stop ═══"; // ═══════════════
input int StopLossPoints = 300;        // Stop Loss Initial (Points)
input bool UseTrailingStop = true;     // Activer Trailing Stop
input int TrailActivation = 200;       // Activation après X points
input int TrailOffset = 100;           // Distance du Suivi (points)

input string Sep4 = "═══ Trading Settings ═══"; // ═══════════════
input double LotSize = 0.01;       // Taille du lot
input int MagicNumber = 12345;     // Magic Number
input int Slippage = 10;           // Slippage maximum

//--- Global Variables
int bbHandle;
double bbUpper[], bbMiddle[], bbLower[];
datetime lastBarTime = 0;
ulong currentTicket = 0;
double highestSinceEntry = 0;
double lowestSinceEntry = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize Bollinger Bands indicator
    bbHandle = iBands(_Symbol, PERIOD_M1, BBPeriod, 0, BBDeviation, BBSource);
    
    if(bbHandle == INVALID_HANDLE)
    {
        Print("Error creating Bollinger Bands indicator");
        return(INIT_FAILED);
    }
    
    // Set arrays as series (newest first)
    ArraySetAsSeries(bbUpper, true);
    ArraySetAsSeries(bbMiddle, true);
    ArraySetAsSeries(bbLower, true);
    
    Print("Dark Venus EA initialized successfully");
    Print("SL: ", StopLossPoints, " points, Trail Activation: ", TrailActivation, 
          " points, Trail Offset: ", TrailOffset, " points");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(bbHandle != INVALID_HANDLE)
        IndicatorRelease(bbHandle);
        
    Print("Dark Venus EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    // Get current prices
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick))
        return;
    
    // Check for new bar (for entry signals)
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    bool isNewBar = (currentBarTime != lastBarTime);
    
    // Get Bollinger Bands values
    if(CopyBuffer(bbHandle, 1, 0, 3, bbUpper) <= 0) return;
    if(CopyBuffer(bbHandle, 0, 0, 3, bbMiddle) <= 0) return;
    if(CopyBuffer(bbHandle, 2, 0, 3, bbLower) <= 0) return;
    
    // Check if we have an open position
    bool hasPosition = CheckOpenPosition();
    
    // Entry logic - only on new bar close
    if(isNewBar && !hasPosition)
    {
        lastBarTime = currentBarTime;
        
        // Get previous candle close
        double prevClose = iClose(_Symbol, PERIOD_M1, 1);
        double prevBBUpper = bbUpper[1];
        double prevBBLower = bbLower[1];
        
        // Check entry conditions
        bool canLong = (TradeDirection == LONG_ONLY || TradeDirection == BOTH);
        bool canShort = (TradeDirection == SHORT_ONLY || TradeDirection == BOTH);
        
        if(BBStrategy == SELL_ABOVE_BUY_BELOW)
        {
            // Buy when close below lower band
            if(prevClose < prevBBLower && canLong)
            {
                OpenPosition(ORDER_TYPE_BUY);
            }
            // Sell when close above upper band
            else if(prevClose > prevBBUpper && canShort)
            {
                OpenPosition(ORDER_TYPE_SELL);
            }
        }
        else // BUY_ABOVE_SELL_BELOW
        {
            // Buy when close above upper band
            if(prevClose > prevBBUpper && canLong)
            {
                OpenPosition(ORDER_TYPE_BUY);
            }
            // Sell when close below lower band
            else if(prevClose < prevBBLower && canShort)
            {
                OpenPosition(ORDER_TYPE_SELL);
            }
        }
    }
    else if(!isNewBar)
    {
        lastBarTime = currentBarTime;
    }
    
    // Exit logic - check on every tick
    if(hasPosition)
    {
        ManagePosition(tick);
    }
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                  |
//+------------------------------------------------------------------+
bool CheckOpenPosition()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionSelectByTicket(ticket))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetString(POSITION_SYMBOL) == _Symbol)
            {
                currentTicket = ticket;
                return true;
            }
        }
    }
    currentTicket = 0;
    return false;
}

//+------------------------------------------------------------------+
//| Open a new position                                                |
//+------------------------------------------------------------------+
bool OpenPosition(ENUM_ORDER_TYPE orderType)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    double price, sl;
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    
    if(orderType == ORDER_TYPE_BUY)
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        sl = price - StopLossPoints * point;
        highestSinceEntry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    }
    else
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        sl = price + StopLossPoints * point;
        lowestSinceEntry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    }
    
    // Normalize prices
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    price = NormalizeDouble(price, digits);
    sl = NormalizeDouble(sl, digits);
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = LotSize;
    request.type = orderType;
    request.price = price;
    request.sl = sl;
    request.tp = 0;  // No take profit - we use trailing stop
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "DarkVenus";
    request.type_filling = ORDER_FILLING_IOC;
    request.type_time = ORDER_TIME_GTC;
    
    if(!OrderSend(request, result))
    {
        Print("OrderSend error: ", GetLastError(), " RetCode: ", result.retcode);
        return false;
    }
    
    if(result.retcode == TRADE_RETCODE_DONE)
    {
        Print("Position opened: ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"),
              " @ ", price, " SL: ", sl);
        currentTicket = result.order;
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Manage open position (trailing stop logic)                         |
//+------------------------------------------------------------------+
void ManagePosition(MqlTick &tick)
{
    if(!PositionSelectByTicket(currentTicket))
        return;
    
    double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentSL = PositionGetDouble(POSITION_SL);
    ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    
    double slDistance = StopLossPoints * point;
    double trailActivation = TrailActivation * point;
    double trailOffset = TrailOffset * point;
    
    bool shouldClose = false;
    string closeReason = "";
    double closePrice = 0;
    
    if(posType == POSITION_TYPE_BUY)
    {
        closePrice = tick.bid;
        
        // Update highest price since entry
        if(tick.bid > highestSinceEntry)
            highestSinceEntry = tick.bid;
        
        // Check fixed stop loss
        double fixedSL = entryPrice - slDistance;
        if(closePrice <= fixedSL)
        {
            shouldClose = true;
            closeReason = "Fixed SL";
        }
        
        // Check trailing stop
        if(!shouldClose && UseTrailingStop)
        {
            double profit = highestSinceEntry - entryPrice;
            if(profit >= trailActivation)
            {
                double trailingSL = highestSinceEntry - trailOffset;
                if(closePrice <= trailingSL)
                {
                    shouldClose = true;
                    closeReason = "Trailing Stop";
                }
            }
        }
    }
    else // SELL
    {
        closePrice = tick.ask;
        
        // Update lowest price since entry
        if(tick.ask < lowestSinceEntry)
            lowestSinceEntry = tick.ask;
        
        // Check fixed stop loss
        double fixedSL = entryPrice + slDistance;
        if(closePrice >= fixedSL)
        {
            shouldClose = true;
            closeReason = "Fixed SL";
        }
        
        // Check trailing stop
        if(!shouldClose && UseTrailingStop)
        {
            double profit = entryPrice - lowestSinceEntry;
            if(profit >= trailActivation)
            {
                double trailingSL = lowestSinceEntry + trailOffset;
                if(closePrice >= trailingSL)
                {
                    shouldClose = true;
                    closeReason = "Trailing Stop";
                }
            }
        }
    }
    
    // Close position if needed
    if(shouldClose)
    {
        ClosePosition(closeReason);
    }
}

//+------------------------------------------------------------------+
//| Close the current position                                         |
//+------------------------------------------------------------------+
bool ClosePosition(string reason)
{
    if(!PositionSelectByTicket(currentTicket))
        return false;
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    double volume = PositionGetDouble(POSITION_VOLUME);
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = volume;
    request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = (posType == POSITION_TYPE_BUY) ? 
                    SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                    SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    request.position = currentTicket;
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "DarkVenus Close: " + reason;
    request.type_filling = ORDER_FILLING_IOC;
    
    if(!OrderSend(request, result))
    {
        Print("Close error: ", GetLastError());
        return false;
    }
    
    if(result.retcode == TRADE_RETCODE_DONE)
    {
        double pnl = PositionGetDouble(POSITION_PROFIT);
        Print("Position closed (", reason, ") P&L: ", pnl);
        
        currentTicket = 0;
        highestSinceEntry = 0;
        lowestSinceEntry = 0;
        return true;
    }
    
    return false;
}
//+------------------------------------------------------------------+
