# Project Layout

    .
    ├── data            # Raw data folder
    |    ├── raw        # Folder that should contain boscun-longitudinal.csv to run code
    ├── output          # Output from main.R, creates train/test states & rewards
    ├── results         # Contains training/test episode reward histories
    ├── src             # R code for aggregating raw data
    |   ├── process.R   # R code to convert raw itineraries to aggregated state representations
    |   ├── main.R      # Run code & save results to output/
    ├── environment.py  # Python module for navigating states & calculating rewards
    ├── ddqn.py         # Python script to run DQN on itinerary data
    └── report.Rmd      # Submitted report R Markdown
    
# Running Code  

1. Ensure that the file ```boscun-longitudinal.csv``` is located in ```.data/raw/```  
2. Generate states & rewards from raw data by executing R script ```src/main.R```  
3. Run DQN by executing ```python ddqn.py```  
4. Results generated in ```results/```