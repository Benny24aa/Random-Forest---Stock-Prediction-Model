library(quantmod)
library(TTR)
library(randomForest)
library(dplyr)

# --- Historical data ---
atos  <- getSymbols("ATO.PA", src="yahoo", from="2015-01-01", auto.assign=FALSE)


dummy_data <- FALSE   # set to FALSE if you don’t want to add the row

if (dummy_data) {
  # Example: manually add a row for a specific date
  new_date <- as.Date("2025-09-01")   # choose the date
  
  
  #	
  # ATO.PA.Open
  # ATO.PA.High
  # ATO.PA.Low
  # ATO.PA.Close
  # ATO.PA.Volume
  # ATO.PA.Adjusted
  
  # Your custom values in the correct order
  vals <- c(43.828, 43.99, 40.582, 41.400, 166000, 41.400)
  
  # Create xts row with same structure
  new_row <- xts(matrix(vals, nrow = 1), order.by = new_date)
  colnames(new_row) <- colnames(atos)
  
  # Append to atos
  atos <- rbind(atos, new_row)
}


ClA <- Cl(atos)
HiA <- Hi(atos)
LoA <- Lo(atos)

retA <- dailyReturn(ClA)
Lag1 <- Lag(retA,1)
Lag2 <- Lag(retA,2)

# --- Indicators ---
SMA10 <- SMA(ClA, 10)
SMA50 <- SMA(ClA, 50)
RSI14 <- RSI(ClA, 14)
Vol5 <- runSD(retA,5)
Vol10 <- runSD(retA,10)
ATR14 <- ATR(HLC(atos), n=14)$atr

# Lag High/Low
LagHi1 <- Lag(HiA,1)
LagLo1 <- Lag(LoA,1)

# Target
NextRet_raw <- lead(retA,1)
NextRet <- pmax(pmin(NextRet_raw,0.05), -0.05)

# Dataset
dataset_xts <- na.omit(merge(retA,Lag1,Lag2,
                             SMA10,SMA50,RSI14,
                             LagHi1,LagLo1,Vol5,Vol10,ATR14,
                             NextRet))
colnames(dataset_xts) <- c("Ret","Lag1","Lag2",
                           "SMA10","SMA50","RSI14",
                           "LagHi1","LagLo1","Vol5","Vol10","ATR14",
                           "NextRet")
dataset <- as.data.frame(dataset_xts)

# Train/test
n <- nrow(dataset)
train <- dataset[1:(n-30),]

# RF model
set.seed(123)
rf_ret <- randomForest(
  NextRet ~ Lag1 + Lag2 + SMA10 + SMA50 + RSI14 + LagHi1 + LagLo1 + Vol5 + Vol10 + ATR14,
  data=train, ntree=2000, mtry=5
)

# --- Monte Carlo parameters ---
n_days <- 14
n_paths <- 500   # number of simulated paths

last_row <- tail(dataset,1)
cur_price <- as.numeric(tail(ClA,1))

# Store all paths
mc_prices <- matrix(NA, nrow=n_days, ncol=n_paths)

# --- Monte Carlo simulation ---
for(p in 1:n_paths){
  price <- cur_price
  state <- list(
    Lag1 = last_row$Lag1,
    Lag2 = last_row$Lag2,
    SMA10 = last_row$SMA10,
    SMA50 = last_row$SMA50,
    RSI14 = last_row$RSI14,
    LagHi1 = last_row$LagHi1,
    LagLo1 = last_row$LagLo1,
    Vol5 = as.numeric(tail(Vol5,1)),
    Vol10 = as.numeric(tail(Vol10,1)),
    ATR14 = as.numeric(tail(ATR14,1))
  )
  
  for(d in 1:n_days){
    newdata <- data.frame(
      Lag1 = state$Lag1,
      Lag2 = state$Lag2,
      SMA10 = state$SMA10,
      SMA50 = state$SMA50,
      RSI14 = state$RSI14,
      LagHi1 = state$LagHi1,
      LagLo1 = state$LagLo1,
      Vol5 = state$Vol5,
      Vol10 = state$Vol10,
      ATR14 = state$ATR14
    )
    
    # Stochastic return: predicted RF return + noise proportional to volatility
    mu <- as.numeric(predict(rf_ret, newdata=newdata))
    sigma <- state$Vol5   # short-term volatility
    ret_pred <- rnorm(1, mean = mu, sd = sigma)
    ret_pred <- max(min(ret_pred, 0.05), -0.05)
    
    # Update price
    price <- price * (1 + ret_pred)
    
    # Update state for next step
    state$Lag2 <- state$Lag1
    state$Lag1 <- ret_pred
    state$SMA10 <- (state$SMA10*9 + price)/10
    state$SMA50 <- (state$SMA50*49 + price)/50
    state$Vol5  <- (state$Vol5*4 + abs(ret_pred))/5
    state$Vol10 <- (state$Vol10*9 + abs(ret_pred))/10
    state$ATR14 <- (state$ATR14*13 + abs(ret_pred))/14
    
    delta <- price - (price/(1+ret_pred))
    up <- ifelse(delta>0, delta,0)
    dn <- ifelse(delta<0, -delta,0)
    RSI14 <- 100 - (100/(1 + mean(c(rep(0,13),up))/mean(c(rep(0,13),dn))))
    state$RSI14 <- RSI14
    state$LagHi1 <- price * (1 + rnorm(1,0,state$Vol5))
    state$LagLo1 <- price * (1 - rnorm(1,0,state$Vol5))
    
    mc_prices[d,p] <- price
  }
}

# --- Prepare data for plotting ---
future_dates <- seq(from = tail(index(ClA),1)+1, by="days", length.out=n_days*2)
future_dates <- future_dates[weekdays(future_dates) %in% c("Monday","Tuesday","Wednesday","Thursday","Friday")]
future_dates <- future_dates[1:n_days]

mc_df <- data.frame(Day = 1:n_days, Date = future_dates,
                    Median = apply(mc_prices,1,median),
                    Lower = apply(mc_prices,1,quantile, probs=0.05),
                    Upper = apply(mc_prices,1,quantile, probs=0.95))

# --- Plot ---
library(ggplot2)
ggplot(mc_df, aes(x=Date)) +
  geom_ribbon(aes(ymin=Lower, ymax=Upper), fill="skyblue", alpha=0.3) +
  geom_line(aes(y=Median), color="blue", size=1.2) +
  labs(title="14-Day Monte Carlo Forecast of ATO.PA",
       y="Predicted Price", x="Date") +
  theme_minimal(base_size=14)

library(ggplot2)

# last actual closing price
last_price <- as.numeric(tail(ClA, 1))

# Match the structure of mc_df
anchor_row <- data.frame(
  Date   = tail(index(ClA), 1),
  Median = last_price,
  Upper  = last_price,
  Lower  = last_price
)

# Keep only relevant columns from mc_df
mc_df_plot <- rbind(anchor_row, mc_df[, c("Date","Median","Upper","Lower")])

# Plot
ggplot(mc_df_plot, aes(x = Date)) +
  geom_line(aes(y = Median), color = "blue", size = 1.2) +
  geom_line(aes(y = Upper), color = "red", linetype = "dashed", size = 1) +
  geom_line(aes(y = Lower), color = "green", linetype = "dashed", size = 1) +
  labs(title = "14-Day Monte Carlo Forecast of ATO.PA",
       y = "Predicted Price", x = "Date") +
  theme_minimal(base_size = 14)

library(ggplot2)

library(ggplot2)

 upper_chart <- ggplot(mc_df, aes(x = Date)) +
  geom_line(aes(y = Upper), color = "red", size = 1.2) +
  labs(title = "14-Day Forecast of ATO.PA (Upper Only)",
       y = "Predicted Price", x = "Date") +
  theme_minimal(base_size = 14)

 
 
median_chart <- ggplot(mc_df, aes(x = Date)) +
  geom_line(aes(y = Median), color = "red", size = 1.2) +
  labs(title = "14-Day Forecast of ATO.PA (Upper Only)",
       y = "Predicted Price", x = "Date") +
  theme_minimal(base_size = 14)

lower_chart <- ggplot(mc_df, aes(x = Date)) +
  geom_line(aes(y = Lower), color = "red", size = 1.2) +
  labs(title = "14-Day Forecast of ATO.PA (Upper Only)",
       y = "Predicted Price", x = "Date") +
  theme_minimal(base_size = 14)


flextable_format <- function(data) {
  data %>%
    flextable() |>
    bold(part = "header") %>%
    bg(bg = "#43358B", part = "header") %>%
    color(color = "white", part = "header") %>%
    align(align = "left", part = "header") %>%
    valign(valign = "center", part = "header") %>%
    valign(valign = "top", part = "body") %>%
    colformat_num(big.mark = ",") %>%
    fontsize(size = 8, part = "all") %>%
    font(fontname = "Arial", part = "all") %>%
    border(border = fp_border_default(color = "#000000", width = 0.5), part = "all") |>
    autofit()
}

library(flextable)

data_table <- mc_df %>% 
  flextable_format() %>% 
  set_table_properties("autofit")


############## Upper Forecast

last_14_actuals <- tail(ClA, 14)
actual_df <- data.frame(
  Date   = index(last_14_actuals),
  Actual = as.numeric(last_14_actuals)
)


# --- Anchor last actual to forecast ---
last_point <- actual_df %>% 
  tail(1) %>% 
  transmute(Date, Value = Actual, Type = "Forecast_Upper")


# --- Combined dataset (Actual + Forecast with anchor) ---
combined_df_upper <- bind_rows(
  actual_df %>% rename(Value = Actual) %>% mutate(Type = "Actual"),
  last_point,  # ensures the lines join without gap
  mc_df %>% select(Date, Value = Upper) %>% mutate(Type = "Forecast_Upper")
)

# --- Plot Actual vs Forecast ---
graph_combined_upper <- ggplot(combined_df_upper, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1.2) +
  labs(title = "Atos (ATO.PA): Last 14 Actuals + Next 14-Day Forecast",
       y = "Price", x = "Date") +
  theme_minimal(base_size = 14)


###### Median Forecast

last_point_median <- actual_df %>% 
  tail(1) %>% 
  transmute(Date, Value = Actual, Type = "Forecast_Median")

# --- Combined dataset (Actual + Median Forecast with anchor) ---
combined_df_median <- bind_rows(
  actual_df %>% rename(Value = Actual) %>% mutate(Type = "Actual"),
  last_point_median,
  mc_df %>% select(Date, Value = Median) %>% mutate(Type = "Forecast_Median")
)

# --- Plot Actual vs Median Forecast ---
graph_combined_median <- ggplot(combined_df_median, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1.2) +
  labs(title = "Atos (ATO.PA): Last 14 Actuals + Next 14-Day Forecast (Median)",
       y = "Price", x = "Date") +
  theme_minimal(base_size = 14)

# --- Flextable ---
data_table_median <- combined_df_median %>% 
  flextable_format() %>% 
  set_table_properties("autofit")

#### Lower forecast

# --- Anchor last actual to forecast (Lower) ---
last_point_lower <- actual_df %>% 
  tail(1) %>% 
  transmute(Date, Value = Actual, Type = "Forecast_Lower")

# --- Combined dataset (Actual + Lower Forecast with anchor) ---
combined_df_lower <- bind_rows(
  actual_df %>% rename(Value = Actual) %>% mutate(Type = "Actual"),
  last_point_lower,
  mc_df %>% select(Date, Value = Lower) %>% mutate(Type = "Forecast_Lower")
)

# --- Plot Actual vs Lower Forecast ---
graph_combined_lower <- ggplot(combined_df_lower, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1.2) +
  labs(title = "Atos (ATO.PA): Last 14 Actuals + Next 14-Day Forecast (Lower)",
       y = "Price", x = "Date") +
  theme_minimal(base_size = 14)

data_dir <- "Previous Predictions Data"

# --- Step 1: Find the most recent CSV file ---
csv_files <- list.files(path = data_dir, pattern = "^mc_df_plot_output_.*\\.csv$", full.names = TRUE)

if(length(csv_files) > 0){
  # Extract timestamps from filenames
  file_dates <- as.Date(sub("mc_df_plot_output_(.*)\\.csv", "\\1", basename(csv_files)))
  
  # Get the most recent
  latest_file <- csv_files[which.max(file_dates)]
  
  # Load the most recent previous predictions
  previous_mc_df <- read.csv(latest_file)
  message("Loaded previous predictions from: ", latest_file)
} else {
  previous_mc_df <- NULL
  message("No previous prediction files found.")
}

if(length(csv_files) > 0){
  latest_file <- csv_files[which.max(file_dates)]
  
  previous_mc_df <- read.csv(latest_file) %>%
    mutate(
      Date = as.Date(Date),
      File_Date = as.Date(File_Date)
    )
  
  message("Loaded previous predictions from: ", latest_file)
}



today <- Sys.Date()



mc_df_plot_output <- mc_df_plot %>%
  filter(Date != min(Date)) %>% 
  mutate(File_Date = today)
  

mc_df_plot_output_final <- bind_rows(previous_mc_df, mc_df_plot_output)

# Save CSV with date in filename
file_name <- paste0("Previous Predictions Data/mc_df_plot_output_", today, ".csv")

write.csv(
  mc_df_plot_output_final,
  file = file_name,
  row.names = FALSE
)

mc_df_plot_output_final_cut <- mc_df_plot_output_final %>%
  group_by(Date) %>%
  slice_max(File_Date, n = 1) %>%  # take the row with max File_Date
  ungroup() %>% 
  select(-File_Date)

last_14_actuals_high <- tail(HiA, 14)
high_df <- data.frame(
  Date = index(last_14_actuals_high),
  Actual_High = as.numeric(last_14_actuals_high)
)

last_14_actuals_low <- tail(LoA, 14)
low_df <- data.frame(
  Date = index(last_14_actuals_low),
  Actual_Low = as.numeric(last_14_actuals_low)
)

mc_df_plot_output_final_cut <- mc_df_plot_output_final_cut %>%
  left_join(actual_df, by = "Date") %>%
  left_join(high_df, by = "Date") %>%
  left_join(low_df,  by = "Date")


mc_df_plot_output_final_cut <- mc_df_plot_output_final_cut %>%
  mutate(
    `Median Residual` = Actual - Median,
    `Low Residual`    = Actual_Low - Lower,
    `Upper Residual`  = Actual_High - Upper,
    Coverage = case_when(
      is.na(Actual) ~ "Not Available",
      Actual >= Lower & Actual <= Upper ~ "Inside Interval",
      TRUE ~ "Outside Interval"
    ),
    `Overall Residual` = rowMeans(
      cbind(abs(`Median Residual`), abs(`Low Residual`), abs(`Upper Residual`)),
      na.rm = FALSE
    )
  )
  
  data_table_actual_vs_model <- mc_df_plot_output_final_cut %>%
    flextable_format() %>%
    # For all columns, replace NA with "Not Available" in display
    colformat_double(na_str = "Not Available") %>%  # for numeric columns
    colformat_char(na_str = "Not Available") %>%    # for character columns
    set_table_properties(layout = "autofit")
  
  data_table_actual_vs_model
  
  
  mc_df_plot_output_final_cut_graphs <- na.omit(mc_df_plot_output_final_cut)
  
  
 median_diff_graph <- ggplot(mc_df_plot_output_final_cut_graphs, aes(x = Date, y = 'Median Residual')) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "Median vs Actual Difference", y = "Median - Actual") +
    theme_minimal()
  
  # Lower difference
 lower_diff_graph <- ggplot(mc_df_plot_output_final_cut_graphs, aes(x = Date, y = Lower_Diff)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    labs(title = "Lower vs Actual Difference", y = "Lower - Actual") +
    theme_minimal()
  
  # Upper difference
 upper_diff_graph <- ggplot(mc_df_plot_output_final_cut_graphs, aes(x = Date, y = Upper_Diff)) +
    geom_line(color = "green") +
    geom_point(color = "green") +
    labs(title = "Upper vs Actual Difference", y = "Upper - Actual") +
    theme_minimal()
 
 library(rmarkdown)
 
 #### Code run for report
 # Produces the report
 rmarkdown::render(
   "C:/Users/benny/Documents/My Resps/Random-Forest---Stock-Prediction-Model/Report.Rmd",

   
   
   
   # Naming the output file with date and health board name
   output_file = paste0(
     "C:/Users/benny/Documents/My Resps/Random-Forest---Stock-Prediction-Model/", Sys.Date(), "_",
     "atos.docx"
   ))
 