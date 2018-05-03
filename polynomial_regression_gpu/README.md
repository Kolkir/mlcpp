# Polynomial regression with MShadow library tutorial

Hello, this is my second article about how to use modern C++ for solving machine learning problems. This time I will show how to make a model for polynomial regression problem described in previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression), but now with another library which allows you to use your GPU easily.

For this tutorial I chose [MShadow](https://github.com/dmlc/mshadow) library, you can find documentation for it [here](https://github.com/dmlc/mshadow/tree/master/doc). This library was chosen because it is actively developed now, and used as a basis for one of a wide used deep learning framework [MXNet](https://mxnet.incubator.apache.org/). Also it is a header only library with minimal dependencies, so it's integration is not hard at all.

Please look at previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) to learn how to download and parse training data.

1. **Loading data to MShadow datastructures**

    
2. **Standardization**

   
7. **Generating new data for testing model predictions**

  
8. **Batch gradient descent implementation**

   
9. **Generating additional polynomial components**

 
10. **Creating general regression model**

    
11. **Making predictions**

   
12. **Plot results**

    To plot data I used [plotcpp](https://github.com/Kolkir/plotcpp) library which is thin wrapper for ``gnuplot`` application. This library use iterators for access to plotting data so I needed to adapt ``XTensor`` matrices to objects which can provide STL compatible iterators, ``xt::view`` function returns such objects.
    ``` cpp
    auto x_coord = xt::view(new_x, xt::all());
    auto line = xt::view(line_values, xt::all());
    auto polyline = xt::view(poly_line_values, xt::all());
    ```
    Next I created plot object, configured it and plotted the data and approximation results.
    ``` cpp
    plotcpp::Plot plt(true);
    plt.SetTerminal("qt"); // show ui window with plots
    plt.SetTitle("Web traffic over the last month");
    plt.SetXLabel("Time");
    plt.SetYLabel("Hits/hour");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid"); // show coordinate grid under plots

    // change X axis values interval
    auto time_range = minmax[0][1] - minmax[0][0];
    auto tic_size = 7 * 24;
    auto time_tics = time_range / tic_size;
    plt.SetXRange(-tic_size / 2, minmax[0][1] + tic_size / 2);

    // change X axis points labels to correspond to week duration
    plotcpp::Plot::Tics xtics;
    for (size_t t = 0; t < time_tics; ++t) {
      xtics.push_back({"week " + std::to_string(t), t * tic_size});
    }
    plt.SetXTics(xtics);

    plt.Draw2D(plotcpp::Points(data_x.begin(), data_x.end(), data_y.begin(),
                               "points", "lc rgb 'black' pt 1"),
               plotcpp::Lines(x_coord.begin(), x_coord.end(), line.begin(),
                              "line approx", "lc rgb 'red' lw 2"),
               plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline.begin(),
                              "poly line approx", "lc rgb 'green' lw 2"));
    plt.Flush();
    ```
    With this code we get such plots:
    ![plots](plot.png)

You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).

Next time I will solve this task with [MShadow](https://github.com/dmlc/mshadow) library to expose power of a GPU.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjkzNTk4MTc2LC0xNDQ1MjY2MDc0LDExNT
QyNzExMzgsNDI3MjMxOTM2XX0=
-->
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMjAxOTc2NTcsLTE3Mjk5NzY2NTddfQ
==
-->