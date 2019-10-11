import pyopencl as cl
import pyopencl.array
import numpy as np
import datetime
import matplotlib.pyplot as plt

# this is the version of the code that gets an output for the parallel computation and plots
# this is the final code

class openclModule:
    def __init__(self, idata):
        # idata: an array of random numbers.

        # Get platform and device
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # TODO:
        # Set up a command queue:
        # host variables
        # device memory allocation
        # kernel code

        self.idata = idata # store the random variable that was passed in

        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        self.mf = cl.mem_flags
        self.idata_buffer = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.idata) # create a buffer for the input data
        self.output_buffer = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.idata.nbytes) # create the output buffer


        self.kernel_code = """
        __kernel void compute_fin_diff(__global float* input_data, __global float* output_data) 
        {

            int length_of_array = 0;
            length_of_array = sizeof(input_data)/sizeof(float); // get the length of the input array

            unsigned int index = get_global_id(0); // get the index of the work-item that is currently being worked with

            // calculate the finite difference
            if(index != 0 && index != length_of_array)
            {
                output_data[index - 1] = (input_data[index - 1] + 2*input_data[index] + input_data[index + 1]) / 4;
            }
        }
        """

    def runAdd_parallel(self):
        # return: an array containing the finite differences from idata and running time.
        # TODO:
        # function call
        # memory copy to host
        # Return output and measured time

        # copy the memory to device
        input_gpu = cl.array.to_device(self.queue, self.idata) # send the input data to the device

        blank_output = np.zeros(len(self.idata) - 2).astype(np.float32) # initialize the output as all zeros
        output_gpu = cl.array.to_device(self.queue, blank_output) # send an initialized empty array to the device

        program = cl.Program(self.ctx, self.kernel_code).build() # compile the program
        program.compute_fin_diff(self.queue, self.idata.shape, None, input_gpu.data, output_gpu.data)

        output_data = output_gpu.get()

        return output_data

    def runAdd_serial(self):
        # return: an array containing the finite differences from idata and running time.
        input_array = self.idata # get the input data
        output = np.zeros(len(self.idata) - 2) # the output will have 2 less indicies as the input

        for i in range(1, len(input_array) -1):

            # calculate the finite difference using python
            output[i-1] = (input_array[i-1] + 2*input_array[i] + input_array[i +1]) / 4

        return output # return the python finite difference

N = 25
py_times = [] 
parallel_times = []

for i in range(1,40):

    random_array = np.random.randint(low=0, high=10, size=N*i).astype(np.float32) # create an array of random numbers
    gpu_array = openclModule(random_array)

    py_start_time = datetime.datetime.now() # record the time right before it runs the serial operation
    py_output = gpu_array.runAdd_serial()
    py_times.extend([(datetime.datetime.now() - py_start_time).total_seconds()]) # calculate how much time the serial operation took

    gpu_start_time = datetime.datetime.now() # record the time right before it ran the operation in parallel
    parallel_output = gpu_array.runAdd_parallel()
    parallel_times.extend([(datetime.datetime.now() - gpu_start_time).total_seconds()]) # calculate how many seconds the gpu calculation took

    print 'py_output=\n', py_output # py_output is the output of your serial function
    print 'parallel_output=\n', parallel_output # parallel_output is the output of your parallel function
    print 'Code equality:\t', py_output==parallel_output

    print 'string_len=', len(random_array), '\tpy_time: ', py_times[i-1], '\tparallel_time: ', parallel_times[i-1] # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.

plt.plot(py_times, label='serial time')
plt.plot(parallel_times, label='parallel time')
plt.legend(fontsize="x-large") # using a named size
plt.show()

