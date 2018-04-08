# Polynomial regression tutorial with XTensor library

1. **Downloading data**

   We use STL ``filesystem`` library to check file existance to prevent multiple downloads, 
   and use libcurl library for downloading see ``utils::DownloadFile`` implementation for details.
    ``` cpp
    const std::string data_path{"web_traffic.tsv"};
    if (!fs::exists(data_path)) {
      const std::string data_url{
          R"(https://raw.githubusercontent.com/luispedro/BuildingMachineLearningSystemsWithPython/master/ch01/data/web_traffic.tsv)"};
      if (!utils::DownloadFile(data_url, data_path)) {
        std::cerr << "Unable to download the file " << data_url << std::endl;
        return 1;
      }
    }
    ```
2. **Parsing data**

    Pay attention on how we process parse exceptions to ignore bad formated items.
    ``` cpp
    io::CSVReader<2, io::trim_chars<' '>, io::no_quote_escape<'\t'>> data_tsv(
      data_path);

    std::vector<DType> raw_data_x;
    std::vector<DType> raw_data_y;

    bool done = false;
    do {
      try {
        DType x = 0, y = 0;
        done = !data_tsv.read_row(x, y);
        if (!done) {
          raw_data_x.push_back(x);
          raw_data_y.push_back(y);
        }
      } catch (const io::error::no_digit& err) {
        // ignore bad formated samples
        std::cout << err.what() << std::endl;
      }
    } while (!done);
    ```
3. **Shuffling data**
    ``` cpp
    size_t seed = 3465467546;
    std::shuffle(raw_data_x.begin(), raw_data_x.end(),
                 std::default_random_engine(seed));
    std::shuffle(raw_data_y.begin(), raw_data_y.end(),
                 std::default_random_engine(seed));
    ```
4. **Loading data to XTensor datastructures**

    We use ``xt::adapt`` function to create views over existent data in ``std::vector`` to prevent data duplicates
    ``` cpp
     size_t rows = raw_data_x.size();
     auto shape_x = std::vector<size_t>{rows};
     auto data_x = xt::adapt(raw_data_x, shape_x);

     auto shape_y = std::vector<size_t>{rows};
     auto data_y = xt::adapt(raw_data_y, shape_y);
    ```
5. MinMax scaling
6. Generate new data for testing model predictions
8. Batch gradient descent implementation
9. Generating additional polynomial components
10. Creating linear model
11. Creating higher order polynomial model
12. Making predictions
13. Plot results
