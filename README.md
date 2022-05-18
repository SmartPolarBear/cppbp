# cppbp

A lightweight extensible library for constructing BP neural network in modern C++.

[![issues](https://img.shields.io/github/issues/SmartPolarBear/cppbp)](https://github.com/SmartPolarBear/cppbp/issues)
[![forks](https://img.shields.io/github/forks/SmartPolarBear/cppbp)](https://github.com/SmartPolarBear/cppbp/fork)
[![stars](https://img.shields.io/github/stars/SmartPolarBear/cppbp)](https://github.com/SmartPolarBear/cppbp/stargazers)
[![license](https://img.shields.io/github/license/SmartPolarBear/cppbp)](https://github.com/SmartPolarBear/cppbp/blob/master/LICENSE)
[![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2F___zirconium___)](https://twitter.com/___zirconium___)

### Built With
#### Environment
This project is built with

- MSVC 17.2.0 and above.
- CMAKE, 3.20 and above.

**Note: Any compiler which supports C++20 may be compatible**

#### Third-party Libraries
- [p-ranav/csv2](https://github.com/p-ranav/csv2)
- [fmtlib/fmt](https://github.com/fmtlib/fmt)
- [google/googletest](https://github.com/google/googletest)

## Usage

### Link cppbp Library

To link cppbp library to your own project, you can include the source tree of cppbp project directly with CMake:  

```cmake
add_subdirectory(cppbp)
```

And then link the library target `cppbp` to some target like:

```cmake
target_link_libraries(<your target> PRIVATE cppbp)
```

### Play With the Example

We provide an example on the classification of the iris dataset. To play with it, build `iris_bp` target. The dataset CSV file should be put in `data/iris.data`, in the same level of directory with the executable file.

## Contributing

Contributions are what make the open source community such an amazing place to be learned, inspire, and create_on_heap. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Furthermore, you are welcomed to:

1. [Ask a question](https://github.com/SmartPolarBear/cppbp/discussions/categories/q-a)   
   Also, have a look at our [FAQs]().
2. [Start a discussion](https://github.com/SmartPolarBear/cppbp/discussions/categories/general)    
   Discussions can be about any topics or ideas related to cppbp.
3. [Make a feature proposal](https://github.com/SmartPolarBear/cppbp/issues)   
   Language features do you want to appear or not to appear in cppbp? For example, you can propose a new grammar making the lox better, or an idea for library features.

## License

Copyright (c) 2022 SmartPolarBear

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.