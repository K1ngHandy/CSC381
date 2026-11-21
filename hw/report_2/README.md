src: https://www.kaggle.com/datasets/samuelpombourcq/us-flight-delays-and-cancellations-2025-2026

About Dataset
US Flight Delays & Cancellations (2025 - 2026)

Context

Every domestic flight operated by the large US airlines from 2025-01-01 to 2026-08-31: 11,730,351 flights, 14 reporting carriers and 361 airports, with scheduled vs actual times, taxi times, delays, delay causes (carrier, weather, air traffic system, security, late aircraft), cancellations with their reason, and diversions. The most-used Kaggle flight datasets stop in 2015, 2018 or 2023; this one covers the most recent 20 months of data the US Bureau of Transportation Statistics (BTS) has published, cleaned into tidy snake_case columns and fast Parquet.

Use it to predict delays or cancellations, rank airlines and airports, study how a late aircraft propagates delay through the day (tail_number), map the network, or explore seasonality (holiday peaks, summer thunderstorms, winter storms).

Year Flights Arrived 15+ min late Cancelled
2025 7,001,619 22.3% 1.47%
2026 4,728,732 23.6% 2.20%
Files

File What it is
flights.parquet All 11,730,351 flights, 40 columns (zstd Parquet; pd.read_parquet loads it in seconds, ~2 GB RAM).
flights_sample.csv A uniform random sample of 500,000 flights (seed 42), same columns, for quick starts and low-memory notebooks.
airlines.csv One row per reporting carrier: name, flight count, on-time arrival rate, cancellation rate.
airports.csv One row per airport: name, city, state, latitude/longitude, elevation, ICAO code, departures and arrivals.
daily_airport_stats.csv Pre-aggregated daily departure statistics per origin airport (202,872 rows): volumes, cancellations, delay rates, delay minutes by cause. Great for time-series forecasting.
Join keys: flights.airline_code = airlines.airline_code; flights.origin / flights.dest = airports.iata_code; daily_airport_stats.airport = airports.iata_code.

Notes on the data

Delay = actual minus scheduled time in minutes; negative values are early. A flight is officially "late" when it arrives 15+ minutes late (arr_del15 = 1).
Cause columns (carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay) are only filled for flights with arr_delay >= 15, and add up to arr_delay for those flights.
Cancelled flights have empty arrival times and delays (a few departed the gate before being cancelled, so they keep a dep_time); diverted flights have empty arrival times and delays.
Times are local hhmm integers (e.g. 1435 = 14:35); 2400 means midnight. Combine flight_date with sched_dep_time to get a timestamp.
Coverage is the BTS Reporting Carrier On-Time Performance table: US carriers with at least 0.5% of domestic scheduled passenger revenue must report, so it covers the large majority of US domestic flights but not every small airline. Regional carriers (SkyWest, Envoy, Endeavor, Republic, PSA …) report under their own code even when flying as American Eagle / Delta Connection / United Express.
Carrier changes inside the period: Hawaiian Airlines (HA) reports through 2025-12-31 only (it was merged into Alaska Airlines' reporting afterwards), and Spirit Airlines (NK) has no flights after 2026-05-01. Keep this in mind when comparing airlines across years.
airline names and cancellation_reason are decoded from the BTS codes. Airport names and coordinates come from OurAirports (matched on IATA code); cities and states come from BTS.
Busiest origin airports: ORD, DEN, ATL, DFW, PHX.
Sources

US Department of Transportation, Bureau of Transportation Statistics, Airline On-Time Performance Data (Reporting Carrier On-Time Performance, 1987-present), monthly files from <https://transtats.bts.gov/PREZIP/> (On_Time_Reporting_Carrier_On_Time_Performance_1987_present_YYYY_M.zip), downloaded 2026-10-28. Field definitions: <https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ>
OurAirports open data (public domain): <https://ourairports.com/data/>
How it was built

Download the 20 monthly files (January 2025 - August 2026; September 2026 is not yet published by BTS).
Keep the 35 columns needed for delay/cancellation analysis (diversion leg details dropped), rename them to snake_case, cast to compact integer types, add year, month, day_of_week, decoded airline and cancellation_reason.
No rows are dropped or deduplicated: one row per flight exactly as reported. Sorted by date, scheduled departure, airline and flight number.
Build the lookup and summary tables from the flights themselves (+ OurAirports for airport names and coordinates).
License

BTS data is a work of the US federal government and is in the public domain; OurAirports data is public domain too. This dataset is released under CC0 1.0. Attribution to BTS is appreciated.

Updates

New months are added as BTS publishes them (usually about two months after the month ends).

Starter ideas

Predict arr_del15 or cancelled from schedule-time features (airline, route, hour, day, month).
Which airline/airport is most reliable? How did a specific storm or holiday weekend ripple through the network?
Delay propagation: follow tail_number through the day and measure how late_aircraft_delay accumulates.
Forecast daily cancellations per airport with daily_airport_stats.csv.
