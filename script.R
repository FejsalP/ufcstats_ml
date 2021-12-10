
################################################################################
################################ DATA SCRAPING #################################
################################################################################

library(rvest)
library(stringr)
library(sjmisc)
library(plyr)

get_list_of_events <- function(ufc_stats_html) {
  table_content <- ufc_stats_html %>%
    html_nodes('i.b-statistics__table-content')
  
  links <- lapply(table_content, function(element) element %>% html_node('a') %>% html_attr('href'))
  return(links)
}

get_event_title <- function(event_html) {
  event_title <- event_html %>%
    html_node('span.b-content__title-highlight') %>%
    html_text()
  event_title <- trimws(event_title)
  return(event_title)
}

get_event_date <- function(event_html) {
  event_date <- event_html %>%
    html_node(xpath = '/html/body/section/div/div/div[1]/ul/li[1]') %>%
    html_text()
  event_date <- trimws(unlist(strsplit(event_date, split = ':', fixed = TRUE))[2])
  return(event_date)
}

get_event_location <- function(event_html) {
  event_location <- event_html %>%
    html_node(xpath = '/html/body/section/div/div/div[1]/ul/li[2]') %>%
    html_text()
  event_location <- trimws(unlist(strsplit(event_location, split = ':', fixed = TRUE))[2])
  return(event_location)
}

get_fights_links_list <- function(event_html) {
  fight_links_list <- event_html %>%
    html_nodes('tr.b-fight-details__table-row') %>%
    html_attr('onclick')
  fight_links_list <- fight_links_list[!is.na(fight_links_list)]
  fight_links_list <- unlist(lapply(fight_links_list, function(element) unlist(strsplit(element, split = "'", fixed = TRUE))[2]))
  return(fight_links_list)
}

get_fight_totals <- function(row, suffix) {
  
  total_totals <- row %>%
    html_nodes('p.b-fight-details__table-text') %>%
    html_text()
  
  total_totals <- trimws(total_totals)
  
  totals_list <- lapply(total_totals, function(x) {
    x_elements <- unlist(strsplit(x, split = ' of ', fixed = TRUE))
    if (length(x_elements) > 1) {
      return(x_elements)
    } else {
      return(x_elements[1])
    }
  })
  
  totals_vec <- unlist(totals_list)
  totals_vec <- ifelse(totals_vec == "---", NA, totals_vec)
  totals_vec[9] <- ifelse(is.na(totals_vec[9]), NA, str_sub(totals_vec[9], 1, -2))
  totals_vec[10] <- ifelse(is.na(totals_vec[10]), NA, str_sub(totals_vec[10], 1, -2))
  totals_vec[19] <- ifelse(is.na(totals_vec[19]), NA, str_sub(totals_vec[19], 1, -2))
  totals_vec[20] <- ifelse(is.na(totals_vec[20]), NA, str_sub(totals_vec[20], 1, -2))
  data_values <- totals_vec[3:length(totals_vec)]
  data_names <- c('red_kd', 
                  'blue_kd', 
                  'red_sig_str', 
                  'red_sig_str_attempt', 
                  'blue_sig_str', 
                  'blue_sig_str_attempt', 
                  'red_sig_str_prec', 
                  'blue_sig_str_prec',
                  'red_tot_str',
                  'red_tot_str_attempt',
                  'blue_tot_str',
                  'blue_tot_str_attempt',
                  'red_td',
                  'red_td_attempt',
                  'blue_td',
                  'blue_td_attempt',
                  'red_td_acc',
                  'blue_td_acc',
                  'red_sub_attempt',
                  'blue_sub_attempt',
                  'red_rev',
                  'blue_rev',
                  'red_ctrl_time',
                  'blue_ctrl_time')
  data_names <- lapply(data_names, function(x) paste(x, suffix, sep = "_"))
  
  if (sum(is.na(data_values)) == length(data_values)) {
    data_values <- rep(NA, length(data_names))
    names(data_values) <- data_names
    return(data_values)
    
  }
  
  names(data_values) <- data_names
  
  return(data_values)
}

get_fight_totals_datarows <- function(fight_html) {
  total_totals_row <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/section[2]/table/tbody/tr')
  
  rounds_table <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/section[3]/table')
  
  rounds_rows <- rounds_table %>%
    html_nodes('tr.b-fight-details__table-row')
  rounds_rows <- rounds_rows[-1]
  
  totals_tot <- get_fight_totals(total_totals_row, "tot")
  totals_rds <- lapply(seq(1, length(rounds_rows)), function(counter, vec) get_fight_totals(vec[counter], paste("rnd", toString(counter), sep = "_")), vec = rounds_rows)
  
  return(unlist(c(list(totals_tot), totals_rds)))
}

get_fight_sigstr <- function(row, suffix) {
  data_list <- row %>%
    html_nodes('p.b-fight-details__table-text') %>%
    html_text()
  
  data_list <- trimws(data_list)
  data_list <- lapply(data_list, function(x) {
    x_elements <- unlist(strsplit(x, split = ' of ', fixed = TRUE))
    if (length(x_elements) > 1) {
      return(x_elements)
    } else {
      return(x_elements[1])
    }
  })
  
  data_vec <- unlist(data_list)
  data_vec <- ifelse(data_vec == "---", NA, data_vec)
  data_vec[7] <- ifelse(is.na(data_vec[7]), NA, str_sub(data_vec[7], 1, -2))
  data_vec[8] <- ifelse(is.na(data_vec[8]), NA, str_sub(data_vec[8], 1, -2))
  data_vec <- data_vec[9:length(data_vec)]
  
  data_names <- c('red_sig_str_head',
                  'red_sig_str_head_attempt',
                  'blue_sig_str_head',
                  'blue_sig_str_head_attempt',
                  'red_sig_str_body',
                  'red_sig_str_body_attempt',
                  'blue_sig_str_body',
                  'blue_sig_str_body_attempt',
                  'red_sig_str_leg',
                  'red_sig_str_leg_attempt',
                  'blue_sig_str_leg',
                  'blue_sig_str_leg_attempt',
                  'red_sig_str_dist',
                  'red_sig_str_dist_attempt',
                  'blue_sig_str_dist',
                  'blue_sig_str_dist_attempt',
                  'red_sig_str_clinch',
                  'red_sig_str_clinch_attempt',
                  'blue_sig_str_clinch',
                  'blue_sig_str_clinch_attempt',
                  'red_sig_str_gnd',
                  'red_sig_str_gnd_attempt',
                  'blue_sig_str_gnd',
                  'blue_sig_str_gnd_attempt')
  data_names <- lapply(data_names, function(x) paste(x, suffix, sep = "_"))
  if (sum(is.na(data_vec)) == length(data_vec)) {
    data_vec <- rep(NA, length(data_names))
    names(data_vec) <- data_names
    return(data_vec)
    
  }
  names(data_vec) <- data_names
  return(data_vec)
}

get_fight_sigstr_datarows <- function(fight_html) {
  sigstr_totals_row <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/table/tbody/tr')
  
  rounds_table <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/section[5]/table')
  rounds_rows <- rounds_table %>%
    html_nodes('tr.b-fight-details__table-row')
  rounds_rows <- rounds_rows[-1]
  
  sigstr_tot <- get_fight_sigstr(sigstr_totals_row, "tot")
  sigstr_rds <- lapply(seq(1, length(rounds_rows)), function(counter, vec) get_fight_sigstr(vec[counter], paste("rnd", toString(counter), sep = "_")), vec = rounds_rows)
  
  return(unlist(c(sigstr_tot, sigstr_rds)))
}

main_fight_details_formatting <- function(x) {
  x <- trimws(unlist(strsplit(x, split = ':', fixed = TRUE))[-1])
  if (length(x) == 1) {
    return(x)
  } else {
    return(paste(x[1], x[2], sep = ":"))
  }
}

get_main_fight_details <- function (fight_html) {
  fighter_containers <- fight_html %>%
    html_nodes('div.b-fight-details__person')
  
  fighter_data <- lapply(fighter_containers, function(element) {
    name <- element %>%
      html_node('a.b-fight-details__person-link') %>%
      html_text()
    name <- trimws(name)
    
    result <- element %>%
      html_node('i.b-fight-details__person-status') %>%
      html_text()
    result <- trimws(result)
    
    nickname <- element %>%
      html_node('p.b-fight-details__person-title')
    
    if (!is.na(nickname)) {
      nickname <- trimws(nickname %>% html_text())
      nickname <- str_sub(nickname, 2, -2)
    }
    
    return(c(name, nickname, result))
  })
  
  division <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/div[2]/div[1]/i') %>%
    html_text()
  division <- trimws(division)
  #division <- unlist(strsplit(trimws(division), split = ' ', fixed = TRUE))[1]
  
  details_container <- fight_html %>%
    html_node(xpath = '/html/body/section/div/div/div[2]/div[2]/p[1]')
  
  details_method <- details_container %>%
    html_node('i.b-fight-details__text-item_first') %>%
    html_text()
  
  details_info <- details_container %>%
    html_nodes('i.b-fight-details__text-item') %>%
    html_text()
  
  method <- main_fight_details_formatting(details_method)
  details_info <- lapply(details_info, main_fight_details_formatting)
  
  df <- c(red = fighter_data[[1]][1], 
          blue = fighter_data[[2]][1], 
          red_nickname = fighter_data[[1]][2], 
          blue_nickname = fighter_data[[2]][2],
          winner = ifelse(fighter_data[[1]][3] == "W", "red", "blue"),
          division = division,
          win_method = method,
          win_round = details_info[[1]][1],
          win_time = details_info[[2]][1],
          time_format = details_info[[3]][1],
          referee = details_info[[4]][1]
  )
  return(df)
}

get_fight_dataframe <- function(fight_link) {
  fight_html <- xml2::read_html(fight_link)
  main_vec <- get_main_fight_details(fight_html)
  totals_vec <- get_fight_totals_datarows(fight_html = fight_html)
  sigstr_vec <- get_fight_sigstr_datarows(fight_html = fight_html)
  
  df_col_names <- c(names(main_vec), names(totals_vec), names(sigstr_vec))
  df_col_values <- c(main_vec, totals_vec, sigstr_vec)
  df <- data.frame(df_col_values)
  df <- as.data.frame(t(df))
  rownames(df) <- c(1)
  colnames(df) <- df_col_names
  return(df)
  
}

scrape_single_event <- function(event_link) {
  event_html <- xml2::read_html(event_link)
  
  event_title <- get_event_title(event_html)
  event_date <- get_event_date(event_html)
  event_location <- get_event_location(event_html)
  fights_links_list <- get_fights_links_list(event_html)
  
  df <- NULL
  for(link in fights_links_list) {
    temp_df <- get_fight_dataframe(link)
    temp_df <- cbind(event_title = event_title, event_date = event_date, event_location = event_location, temp_df)
    
    if (is.null(df)) {
      df <- temp_df
    } else {
      df <- rbind.fill(df, temp_df)
    }
  }
  return(df)
}

scrape_data <- function(events_links) {
  df <- NULL
  counter <- 1
  for(link in events_links) {
    print(paste("Scraping: ", toString(counter), " -> ", link))
    temp_df <- scrape_single_event(link)
    if (is.null(df)) {
      df <- temp_df
    } else {
      df <- rbind.fill(df, temp_df)
    }
    counter <- counter + 1
  }
  return(df)
}

page_url <- 'http://ufcstats.com/statistics/events/completed?page=all'
ufc_stats <- xml2::read_html(page_url)

events_links <- get_list_of_events(ufc_stats_html = ufc_stats)[-1]

df <- scrape_data(events_links)
df2 <- df[,1:302]

write.csv(df2, 'ufcstats_new2.csv', row.names = FALSE)

################################################################################
################################ DATA CLEANING #################################
################################################################################

setwd("C:\\Users\\niX\\Desktop\\Projects\\ufcstats_ml")
df2 <- read.csv("ufcstats_new.csv", header = TRUE, sep = ",")

# Drop the columns that are listed in the vector
clean_df <- subset(df2, select = -c(event_title, event_date, event_location,
                                    red, blue, red_nickname, blue_nickname,
                                    division, referee, win_round, win_time))

# Select just rows which have win_method as some kind of "decision"
clean_df <- clean_df[clean_df$win_method == "Decision - Unanimous" | clean_df$win_method == "Decision - Split" | clean_df$win_method == "Decision - Majority",]

clean_df <- subset(clean_df, select = -c(win_method))

# Drop all fights that are not in "3 Rnd (5-5-5)" time format.
clean_df <- clean_df[clean_df$time_format == "3 Rnd (5-5-5)", ]

# Drop "time_format" column because we don't need it for ML algorithms.
clean_df <- subset(clean_df, select = -c(time_format))

# Drop all columns that hold information about rounds 4 and 5.
clean_df <- clean_df[, !grepl("4|5", colnames(clean_df))]

#Drop all columns that hold information in percentages and rev column.
clean_df <- clean_df[, !grepl("_prec_|_acc_|_rev_", colnames(clean_df))]

# Function that transforms control time from "min:sec" format to number of seconds.
transform_ctrl_time <- function(vec) {
  vec[vec == "--"] <- "0:00"
  list_split <- strsplit(vec, ":")
  lapply(list_split, function(x) {
    return(as.numeric(x[1]) * 60 + as.numeric(x[2]))
  })
}

# Applying transformation of control time to the data frame
clean_df[grep("_ctrl_time_", colnames(clean_df))] <- lapply(clean_df[grep("_ctrl_time_", colnames(clean_df))], transform_ctrl_time)

# Converting character columns to factors
clean_df$winner <- as.factor(clean_df$winner)

# Converting all other columns to numeric data type
clean_df[, -c(1:2)] <- sapply(clean_df[, -c(1:2)], as.numeric)

write.csv(clean_df, 'ufcstats_cleaned.csv', row.names = FALSE)

clean_df <- clean_df[, -c(2:19)]
























