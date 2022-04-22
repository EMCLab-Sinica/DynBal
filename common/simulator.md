# Intermittent CNN simulator on PC

A simulator for running Stateful NN on PC.

## Building

To build it, basic toolchain for C/C++ development is necessary:

* CMake >= 2.8.12
* A modern compiler (gcc or clang) supporting C++ 14

After transforming the model with `./dnn-models/transform.py`, the simulator can be configured with `cmake -B build -S .`

Additional CMake options are available:

* `-D MY_DEBUG=(1|2|3)`: override `MY_DEBUG` macro in `common/my_debug.h`
* `-D USE_PROTOBUF=ON`: enable the `-s` feature described below; needs to install the protobuf library on the system.

After configuring the simulator, it can be built with `make -C build`. The built program can be run with `./build/intermittent-cnn`.

Additional program options are available:

* `-r`: Turns the simulated NVM (`nvm.bin`) into read-only memory (discards any changes). This is useful if you are debugging some issue and want to keep NVM in a specific state.
* `-f`: Dump floating point numbers instead integers during debugging.
* `-c`: Force shutdown after writing N bytes to NVM. This option is used by the `--shutdown-after-writes` of the `run-intermittently.py` script.
* `-s`: Save output features maps to a file instead of printing them out; requires building the simulator with `-D USE_PROTOBUF=ON` and `-D MY_DEBUG=2` or higher.

## Checking inference accuracy

Output feature maps from the Intermittent CNN simulator and ONNX runtime Python library can be saved to .pb files using `-s` and `--save-file`, respectively.
Those .pb files can then be compared to identify possible accuracy issues. For example,

```
$ ./dnn-models/transform.py --target msp432 --ideal har
$ cmake -B build -S . -D MY_DEBUG=2 -D USE_PROTOBUF=ON
$ make -C build
$ ./build/intermittent-cnn -r 1 -s ~/tmp/har-cpp.pb
$ python exp/original_model_run.py --save-file ~/tmp/har-py.pb --limit 1 har
$ python exp/compare-model-output.py --baseline ~/tmp/har-py.pb --target ~/tmp/har-cpp.pb --topk 10
```

In this example, `exp/compare-model-output.py` compares output feature maps in `~/tmp/har-cpp.pb` against `~/tmp/har-py.pb` and lists values with top 10 relative errors in each OFM.

## Checking intermittent execution

There are two approaches to check correctness of intermittent execution - power failures triggered periodically with a given time interval, or forced power failures when a given count of NVM writes is reached.

`exp/run-intermittently.py` handles both approaches. For the first approach, `--interval` can be used. It's recommended to build the simulator with the higest debug level (e.g., `-D MY_DEBUG=3`) so that there are many power failures in each end-to-end inference. For example,

```
$ cmake -B build -S . -D MY_DEBUG=3
$ make -C build
$ rm -vf nvm.bin && python ./exp/run-intermittently.py  --rounds 100 --interval 0.02 ./build/intermittent-cnn
```

For the second approach, `--shutdown-after-writes` can be used. A shell loop can be combined with this approach to test all possible power failures. For example,

```
$ for i in {1..100000..10}; do echo $i; rm -vf nvm.bin && python exp/run-intermittently.py ./build/intermittent-cnn --rounds 1 --shutdown-after-writes $i ; done
```

For both approach, log files are available as `/tmp/intermittent-cnn-*`.
