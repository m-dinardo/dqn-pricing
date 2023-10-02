library(data.table)
library(zoo)

INPUT_DATA_DIR <- 'data/raw/'
DATA <- 'boscun-longitudinal.csv'

boscun_init <- function() {
  # Aggregates itineraries into daily summary of a particular trip
  #
  # Returns:
  #   Sample statistics & features of itineraries grouped by specific trip date / query date pair
  boscun <- fread(paste(INPUT_DATA_DIR, DATA, sep=''))
  # Reduce number of major carriers and create an 'other' category
  boscun$al_group <- factor(regroup_airlines(boscun$major_carrier_id))
  # Create stops grouping
  #boscun$stop_group <- factor(regroup_total_stops(boscun$total_stops))
  prices <- boscun %>% group_by(received_date, departure_odate, return_odate, al_group) %>%
            summarise(min_price = min(total_usd), mean_price = mean(total_usd),
                      sd_price = sd(total_usd), mean_seats = mean(available_seats), tot_itin = n())
  # Set bundles with 1 itinerary to 0 instead of NA sd
  prices$sd_price <- ifelse(is.na(prices$sd_price), 0, prices$sd_price)
  df <- NULL
  # Calculate min, mean, sd price and mean seats, total itineraries for
  # All groupes of airlines for each trip/received date pair
  for(col in c('min_price', 'mean_price', 'sd_price', 'mean_seats', 'tot_itin')) {
    tmp <- prices %>% select_('received_date', 'departure_odate', 'return_odate',
                                'al_group', col) %>% spread_('al_group', col)
    if(col == 'tot_itin') tmp[is.na(tmp)] <- 0
    colnames(tmp)[-(1:3)] <- sapply(colnames(tmp)[-(1:3)], FUN=function(x) {
      paste(x, col, sep='_')
    })
    if(is.null(df)) {
      df <- tmp
    } else {
      df <- bind_cols(df, tmp[-(1:3)])
    }
  }
  df <- all_airlines(df, boscun) # Calculate same stats as above with all airlines
  df <- fill_data(df) # Replace missing airline data with data from ALL group
  df <- filter_low_frq(df) # Remove trips with too few received_dates
  df <- join_features(boscun, df) # Add on snapshot features (length of stay, etc)
  df <- interpolate(df) # Fill in gaps between received dates
  df <- df %>% arrange(trip, received_date) # Order data frame
  df <- lag_features(df) # Add lagged features
  # Calculate daily return (reward & target variable)
  df$daily_return <- (df$ALL_min_price - df$next_min_price) * (df$next_min_price != 0)
  #TODO: Maybe remove daily return outliers?
  return(df)
}

lag_features <- function(trips) {
  # Joins the aggregated features of a trip with its lags
  #
  # Args:
  #   trips: Data frame of aggreagted itineraries
  lag1 <- trips %>% select(-c(advance, length_of_stay, includes_saturday_night_stay)) %>%
    group_by(trip) %>% mutate_at(vars(-received_date), lag, n=1)
  lag3 <- trips %>% select(-c(advance, length_of_stay, includes_saturday_night_stay)) %>%
    group_by(trip) %>% mutate_at(vars(-received_date), lag, n=3)
  
  trips <- trips %>% group_by(trip) %>% mutate(next_min_price = lead(ALL_min_price, 1))
  trips[is.na(trips)] <- 0
  trips <- merge(trips, lag1, by=c('trip', 'received_date'), suffixes = c('','lag1'))
  trips <- merge(trips, lag3, by=c('trip', 'received_date'), suffixes = c('','lag3'))
  trips <- na.omit(trips)
  trips_count <- trips %>% group_by(trip) %>% summarise(n=n()) %>% filter(n > 2)
  trips <- trips %>% filter(trip %in% trips_count$trip)
  return(trips)
}
join_features <- function(bc, trips) {
  # Joins non-aggregate features (days out, length of stay, available seats, etc)
  #
  # Args:
  #   bc: Data frame of raw itineraries with original features
  #   trips: Data frame with grouped summary features of trip/query date combinations
  #
  # Returns:
  #   Data frame with both aggregated features and snapshot features
  bc <- bc %>% unite(trip, departure_odate, return_odate)
  x <- bc %>% select(trip, received_date, advance, length_of_stay, 
                     includes_saturday_night_stay) %>% distinct(trip, received_date, .keep_all = TRUE)
  tot <- merge(trips, x, by=c('trip', 'received_date'), all.x = TRUE)
  return(tot)
}

interpolate <- function(trips) {
  # Fills in gaps between received_data queries with linear interpolation of all features
  #
  # Args:
  #   trips:  data frame with aggregated features
  # 
  # Returns:
  #   trips:  data frame filled with additional received_data rows linearly interpolated
  for(t in unique(trips$trip)) {
    tmp <- trips %>% filter(trip == t)
    min_received <- as.Date(min(tmp$received_date))
    max_received <- as.Date(max(tmp$received_date))
    range <- seq.Date(min_received, max_received, 1)
    missing <- range[!(range %in% as.Date(tmp$received_date))]
    if(length(missing) > 0) {
      tmp_new <- data.frame(received_date = as.character(missing), trip = t)
      trips <- bind_rows(trips, tmp_new)
    }
  }
  trips <- trips[order(trips$trip, trips$received_date),]
  for(col in colnames(trips)[-(1:2)]) {
    trips[,col] <- na.approx(trips[,col])
  }
  return(trips)
}

filter_low_frq <- function(trips) {
  # Remove trips with too few received_date queries, insufficient data
  #
  # Args:
  #   trips: data frame with aggregated features
  #
  # Returns:
  #   trips: with trips removed that have lower than 25% frequency
  #          between the initial and final query
  frq <- trips %>% group_by(departure_odate, return_odate) %>% 
    summarise(min_r = min(received_date), max_r = max(received_date), tot_it = n())
  frq$days_between <- (as.Date(frq$max_r) - as.Date(frq$min_r)) + 1
  frq$itin_prop <- frq$tot_it/as.numeric(frq$days_between)
  frq <- frq %>% filter(itin_prop >= 0.25 & days_between > 1) %>% unite(trip, departure_odate, return_odate)
  
  trips <- trips %>% unite(trip, departure_odate, return_odate) %>%
        filter(trip %in% frq$trip)
  return(trips)
}

fill_data <- function(df) {
  # Replace missing airline data with ALL aggregated data
  # 
  # Args:
  #   df: aggregated data containing missing stats for some airlines
  #      when data is not available
  # Returns:
  #   df: aggregated data with missing airline data filled in
  cols <- c('min_price', 'mean_price', 'sd_price', 'mean_seats')
  als <- c('UA', 'Other')
  for(col in cols) {
    for(al in als) {
      acol <- paste(al, col, sep='_')
      rep <- paste("ALL", col, sep='_')
      df[is.na(df[,acol]), acol] <- df[is.na(df[,acol]), rep]
    }
  }
  return(df)
}

all_airlines <- function(df, boscun) {
  # Calculate summary statistics for ALL airlines
  #
  # Args:
  #   df  data frame with already aggregated-by-airline data
  #   boscun data frame with all itineraries before aggregation
  #
  # Returns:
  #   df  data frame with ALL aggregation, which aggs over all itineraries
  prices_all <- boscun %>% group_by(received_date, departure_odate, return_odate) %>%
    summarise(min_price = min(total_usd), mean_price = mean(total_usd),
              sd_price = sd(total_usd), mean_seats = mean(available_seats), tot_itin = n())
  prices_all$sd_price <- ifelse(is.na(prices_all$sd_price), 0, prices_all$sd_price)
  colnames(prices_all)[-(1:3)] <- sapply(colnames(prices_all)[-(1:3)], FUN=function(x) {
    paste('ALL', x, sep='_')
  })
  df <- bind_cols(df, prices_all[-(1:3)])
  return(df)
}

regroup_airlines <- function(airlines) {
  # Group airlines into two buckets: UA and Other
  #
  # Args:
  #   airlines: character vector of airlines to be processed
  #
  # Returnes:
  #   airlines: factor vector of regrouped airline labels
  case_when(
    #airlines %in% c('AA', 'DL') ~ 'AA/DL',
    airlines == 'UA' ~ 'UA',
    TRUE ~ "Other"
  )
}
