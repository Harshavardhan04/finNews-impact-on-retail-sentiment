#!/bin/bash

# Define the tickers
tickers=('AAPL' 'ABBV' 'ABT' 'ACN' 'ADBE' 'ADP' 'AMAT' 'AMD' 'AMGN' 'AMT' 'AMZN' 'APH' 'AVGO' 'AZO' 'BA' 'BAC' 'BDX' 'BMY' 'C' 'CB' 'CHTR' 'CMCSA' 'COP' 'COST' 'CSCO' 'CVS' 'CVX' 'DELL' 'DHR' 'DIS' 'DUK' 'ED' 'GE' 'GILD' 'GIS' 'GOOGL' 'HD' 'IBM' 'INTC' 'INTU' 'ISRG' 'JNJ' 'JPM' 'KHC' 'KO' 'KR' 'LLY' 'LMT' 'MA' 'MDLZ' 'MDT' 'META' 'MMM' 'MO' 'MRK' 'MSFT' 'MSI' 'NEE' 'NEM' 'NFLX' 'NKE' 'NOC' 'NVDA' 'ORCL' 'PAYX' 'PEP' 'PFE' 'PG' 'PGR' 'PM' 'PSA' 'PYPL' 'QCOM' 'TMO' 'TMUS' 'TSLA' 'TXN' 'UNH' 'UPS' 'V' 'VRTX' 'VZ' 'WCN' 'WFC' 'WMT' 'XOM' 'YUM')

# Input file
input_file="wallstreetbets_comments"

# Output directory
output_dir="ticker_files"
mkdir -p "$output_dir"

# Get the number of CPU cores on macOS
cpu_cores=$(sysctl -n hw.ncpu)

# Ensure GNU grep is installed
if ! command -v grep &>/dev/null; then
    echo "Error: 'grep' is required but not installed. Install it using 'brew install grep'."
    exit 1
fi

# Process each ticker in parallel
echo "${tickers[@]}" | tr ' ' '\n' | xargs -P "$cpu_cores" -I {} sh -c 'grep --line-buffered "{}" '"$input_file"' > '"$output_dir"'/{}.txt'

echo "Processing complete! Files saved in $output_dir"
