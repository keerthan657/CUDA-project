# CUDA-project
Parallelization of enhanced run-length encoding using CUDA

## Run-length encoding
A kind of compression algorithm, which takes sequences of repeated values and compresses them into value and count pair.
- Lossless compression
- Works good on images which are not too complex

## Pairing function
In RLE, we have the (value,count) pair in the result.
But what if we could represent these 2 values in a single unique value.
Having it unique means gettting back the same 2 values while doing the reverse.

Such a mathematical function exists, and is called a pairing function.
It uniqualy encodes two numbers into one single number.
The one used in this project is Szudzik's pairing function.

The link referenced for the formulae: http://szudzik.com/ElegantPairing.pdf

## Steps
- Load image using OpenCV
- Read pixel values and perform RLE
- Do enhanced compression using pairing function
- Utilize GPU to do it parallelly
- Show statistics of compression

## Commands
- To compile the program
```nvcc <program_file> <arguments>```
- To run the program
```./a.out <image_file>```
- OpenCV arguments
```-lopencv_core -lopencv_imgcodecs```

### ```nvcc main.cu -lopencv_core -lopencv_imgcodecs && ./a.out image.jpg```

## Credits
Credits to [adolfos94](https://github.com/adolfos94)
for providing such a wonderful respoitory from which this project was referenced.

Repository link: [Enhanced-Run-Length-Encoding](https://github.com/adolfos94/Enhanced-Run-Length-Encoding)

Pairing function: [An Elegant Pairing Function](http://szudzik.com/ElegantPairing.pdf)

Things different in this repository:
- Removed thrust library
- Used OpenCV to load actual image, instead of random values
- Show some statistics
- Used <time.h> to measure time instead of cudaEventRecord()
