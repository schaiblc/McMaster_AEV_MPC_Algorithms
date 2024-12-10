#!/usr/bin/env python3
import rospy
import numpy as np
import sys
from sensor_msgs.msg import Image as ROSImage
import cv2
from cv_bridge import CvBridge
from PIL import Image
import os
import tensorrt as trt
import pycuda.driver as cuda 

import onnx
import onnxruntime as ort

from matplotlib.colors import hsv_to_rgb



class HostDeviceMem():
    def __init__(self, host_mem, device_mem, name):
        self.host= host_mem
        self.device=device_mem
        self.name=name

class VisionModel():
    def __init__(self):

        cuda.init()
        self.device = cuda.Device(0)  
        self.ctx = self.device.make_context()

        self.stream=cuda.Stream()
        engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/FCN.engine")
        self.logger=logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)

        build_engine=rospy.get_param("~build_cv_engine")
    
        if build_engine==True:
            self.create_rt_engine()
        self.load_engine(engine_path) #defines engine attribute from serialized file
        self.context=self.engine.create_execution_context()
        self.bridge=CvBridge()
        

        self.allocate_buffers()
       
       
    
    def create_rt_engine(self):
        log=trt.Logger(trt.Logger.VERBOSE)
        builder=trt.Builder(log) #builds the engine
        
        config=builder.create_builder_config()
        #set cache
        cache=config.create_timing_cache(b"") #b"" indicates to create a new cache
        config.set_timing_cache(cache, ignore_mismatch=False) #mismatch would occur if using a cache that has contains different CUDA device property
        max_workspace= 2 << 30 #2 Gb of memory
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network=builder.create_network(explicit_batch)

        parser=trt.OnnxParser(network,log)
        model_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/fcn-resnet101-11.onnx")

        #The below commented lines can be used to obtain information on the onnx expected inputs
        '''
        model = onnx.load(model_path)
        session=ort.InferenceSession(model_path)
        print("Input tensors:")
        for input_tensor in session.get_inputs():
            print(f"Name: {input_tensor.name}, Shape: {input_tensor.shape}, Type: {input_tensor.type}")
        '''

        with open(model_path, 'rb') as model_file: #parse onnx file, write to network
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        builder.max_batch_size = 1 #max number of samples that inference can be performed on at once
       

        profile= builder.create_optimization_profile()

        profile.set_shape("input", (1, 3, 520, 640), (1, 3, 520, 640), (1, 3, 520, 640)) #image input
        

        config.add_optimization_profile(profile)

        half = rospy.get_param("~use_fp16")
        if half: #lessons precision to increase speed
            config.set_flag(trt.BuilderFlag.FP16)
            
        serialized_network = builder.build_serialized_network(network, config)

        engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/FCN.engine")
        with open(engine_path, "wb") as f:
            f.write(serialized_network)

        print(f"TensorRT engine saved to {engine_path}")

        

    def load_engine(self,engine_path):

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        

    def pre_process(self,img): #PIL image
        
        cv_img=self.bridge.imgmsg_to_cv2(img, desired_encoding='rgb8')
        #(width,height) resize shape format
        cv_img = cv2.resize(cv_img, (640,520), interpolation=cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
        data = (np.asarray(cv_img).astype('float32') / float(255.0) - mean) / stddev  
        # Switch from HWC to to CHW order
        return np.expand_dims(np.moveaxis(data, 2, 0),0)
         

       
        


    def allocate_buffers(self): #allocate memory on cpu and gpu for input and output data transfer between devices
      
        inputs = [] # Host input
        outputs = [] #buffer memory allocated on cpu to retrieve outputs
        bindings = [] #store addresses of each input and outputs memory blocks. Used by tensorrt to store results and gather inputs


        self.names= []
        self.sizes= []
       
        for binding in self.engine: #loop through all input output tensors


            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx)) 
            shape = self.context.get_binding_shape(binding_idx)
            dtype = self.engine.get_binding_dtype(binding)
            if dtype == trt.float32:
                np_dtype = np.float32
            elif dtype == trt.int32:
                 np_dtype = np.int32
            elif dtype == trt.bool:
                np_dtype = np.bool_
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            # Get binding name
            binding_name = self.engine.get_binding_name(binding_idx)
            self.names.append(binding_name)
            # Get binding shape and data type
            binding_shape = self.engine.get_binding_shape(binding_idx)
           

            # Check if the binding is an input or output
            is_input = self.engine.binding_is_input(binding_idx)
            binding_type = "Input" if is_input else "Output"


            
            host_mem=cuda.pagelocked_empty(size, np_dtype)
            device_mem=cuda.mem_alloc(host_mem.nbytes) 
            bindings.append(int(device_mem))

            

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem,device_mem, binding_name))
                   
            else:
                outputs.append(HostDeviceMem(host_mem,device_mem, binding_name))

            #Print info 
            '''
            print(f"Binding Index: {binding_idx}")
            print(f"Binding Name: {binding_name}")
            print(f"Binding Shape: {binding_shape}")
            print(f"Data Type: {np_dtype}")
            print(f"Size (# elements): {size}")
            print(f"Binding Type: {binding_type}")
            print("=" * 40)
            '''

            self.inputs, self.outputs, self.bindings= inputs, outputs, bindings

   

    def __del__(self):
        self.ctx.pop()
        del self.engine
        del self.context
        del self.runtime
       
        return

    def get_palette(self,num_classes):
    # prepare and return palette
        palette = [0] * num_classes * 3

        for hue in range(num_classes):
            if hue == 0: # Background color
                colors = (0, 0, 0)
            else:
                colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))

            for i in range(3):
                palette[hue * 3 + i] = int(colors[i] * 255)

        return palette

    def colorize(self, labels, num_classes):
        # generate colorized image from output labels and color palette
        result_img = Image.fromarray(labels).convert('P', colors=num_classes)
        result_img.putpalette(self.get_palette(num_classes))
        return np.array(result_img.convert('RGB'))

    def visualize_output(self, image, output, classes, num_classes):
        image= np.squeeze(image)
        image= np.reshape(image, (520,640, 3))
        output= np.squeeze(output)
       
        assert(image.shape[0] == output.shape[1] and \
           image.shape[1] == output.shape[2]) # Same height and width
        assert(output.shape[0] == num_classes)

        # get classification labels
        raw_labels = np.argmax(output, axis=0).astype(np.uint8)

        # comput confidence score
        confidence = float(np.max(output, axis=0).mean())
        
        # generate segmented image
        result_img = self.colorize(raw_labels,num_classes)

        
        # generate blended image
        #blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img_float, 0.5, 0)

        result_img = Image.fromarray(result_img)
        #blended_img = Image.fromarray(blended_img)

        return confidence, result_img, raw_labels



    def postprocess(self, input_image, output_image):
        
        classes_path= engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/FCN_classes.txt")
        classes = [line.rstrip('\n') for line in open(classes_path)]
        num_classes = len(classes)

        return self.visualize_output( input_image, output_image,classes, num_classes)
        
        
    
    def inference(self,input_image): #input image= sensormsgs/Image.msg
    
        
            
        '''    
        print("Input Image is: ", self.input_image.shape)
        print("\nInput buffer at [0] shape is ", self.input_buffers[0].shape)
        print("\nBindings: ", self.bindings)
                        
        for i, (input_buffer, gpu_buffer) in enumerate(zip(self.input_buffers, self.gpu_input_memory)):
            # Print input buffer shape
            print(f"Input Buffer [{i}] Shape: {input_buffer.shape}")
            print(f"Input Buffer [{i}] Data Type: {input_buffer.dtype}")
            # Print size of GPU buffer
            print(f"GPU Input Buffer [{i}] Size: {input_buffer.nbytes} bytes")
                
        for i, (output_buffer, gpu_buffer) in enumerate(zip(self.output_buffers, self.gpu_output_memory)):
            # Print output buffer shape
            print(f"Output Buffer [{i}] Shape: {output_buffer.shape}")
            print(f"Output Buffer [{i}] Data Type: {output_buffer.dtype}")
             # Print size of GPU buffer
            print(f"GPU Output Buffer [{i}] Size: {output_buffer.nbytes} bytes")
        '''    
        self.ctx.push()   


        input_image=self.pre_process(input_image)

        input_image_1D=np.ravel(input_image)

        #1) copy input image into host input buffer
        np.copyto(self.inputs[0].host, input_image_1D) 


        #2) copy input image from host input buffer to gpu input buffer. Memory now ready for inference.
        for inp in self.inputs: #iterate through each host,device buffer pair
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)


         
        #specify device address of inputs and outputs to trt.execution__context for inference
        for inp in self.inputs:
            self.context.set_tensor_address(inp.name, int(inp.device))
               

        for out in self.outputs:
            self.context.set_tensor_address(out.name, int(out.device))

            
            #3)execute inference (can also use async_v3 )
        self.context.execute_async_v2(self.bindings, stream_handle=self.stream.handle)

        output_image= []

        #4)Transfer result from GPU (Device) to host (CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) #copy device data into host data for cpu usuage
            if out.name== "out":
                output_image= np.copy(np.array(out.host))
                
        #5) synchronize cuda stream


        output_image = np.reshape(output_image, (1,21,520,640))
        
       

        self.stream.synchronize()
        
        self.ctx.pop()
        
        conf, result_img, raw_labels = self.postprocess(input_image, output_image)

        cv_result= cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
        image_message = self.bridge.cv2_to_imgmsg(cv_result, encoding="bgr8")

        return image_message

        
  

class AEV_Detector():
    def __init__(self):
        rgb_image_topic=rospy.get_param("~rgb_image_topic")
        FCN_topic=rospy.get_param("~fcn_topic")
        self.model=VisionModel()

        rospy.Subscriber(rgb_image_topic,ROSImage,self.rgb_callback, queue_size=1)
        self.pub = rospy.Publisher(FCN_topic, ROSImage, queue_size=1)
       

        

    def rgb_callback(self, data):
        
        image_message= self.model.inference(data)
         
        self.pub.publish(image_message)



def main(args):
    rospy.init_node("FCN", anonymous=True)
    vehicles=AEV_Detector()
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
