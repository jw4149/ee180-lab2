In multithreading optimization, we reach ~96 FPS
We added four barriers, initialized and destroyed in `main.cpp` and mainly used in `sobel_mt.cpp`
`capReady` is used to make sure video capture devices setup are ready in both threads before preceding
`srcReady` ensures memory allocation for top and bottom halves of image is ready
`grayReady` makes sure image halves are turned into grayscale, and `sobelReady` is used to make sure `sobelCalc` finishes in both halves
In header file `sobel_alg.h` we need to declare these barriers are defined externally
When finishing processing two halves of image separately guaranteed by `sobelReady` barrier, we vertically concatenate two matrices
