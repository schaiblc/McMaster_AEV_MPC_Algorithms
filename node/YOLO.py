#!/usr/bin/env python3
import rospy
import numpy as np
import sys
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float32MultiArray, UInt16MultiArray
from f1tenth_simulator.msg import YoloData
import cv2
from cv_bridge import CvBridge
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import tensorrt as trt
import pycuda.driver as cuda 

import onnx
import onnxruntime as ort


#Receive RGB image from camera publisher

image_rows=480
image_cols= 640

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
        self.use_tiny_yolo= rospy.get_param("~use_tiny_yolo")
        self.use_yolov5= rospy.get_param("~use_yolov5")
        self.min_score= rospy.get_param("~yolo_min_score")
        if self.use_tiny_yolo:
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/tiny-yolo3.engine")
        elif self.use_yolov5:
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/Custom_yolo5.engine")
        else:
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/yolo3.engine")
        self.logger=logger = trt.Logger(trt.Logger.WARNING) #set to Logger.Verbose for more detailed debugging
        self.runtime = trt.Runtime(self.logger)
        self.stream=cuda.Stream()

        build_engine=rospy.get_param("~build_cv_engine")
        self.max_nbox=rospy.get_param("~max_yolo_boxes")

        if build_engine==True:
            self.create_rt_engine()
        
        self.load_engine(engine_path)
        self.context=self.engine.create_execution_context()
        self.bridge=CvBridge()

        
        self.allocate_buffers()
        if self.use_yolov5:
            self.classes= ['car']
        else:

            classes_path= os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/yolo3Classes.txt")
            self.classes = [line.rstrip('\n') for line in open(classes_path)]
        
        
    def create_rt_engine(self):
       
        if self.use_tiny_yolo:
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/tiny-yolo3.engine")
            model_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/tiny-yolov3-11.onnx")


        elif self.use_yolov5:
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/Custom_yolo5.engine")
            model_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/yolov5CustomTrained.onnx")
            
        else:    
            engine_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/yolo3.engine")
            model_path=os.path.expanduser("~/catkin_ws/src/f1tenth_simulator/learning_models/yolov3.onnx")

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
        
        '''

        #The below lines can be used to obtain information on the onnx expected inputs
        model = onnx.load(model_path)
        session=ort.InferenceSession(model_path)
        print("Input tensors:")
        for input_tensor in session.get_inputs():
            print(f"Name: {input_tensor.name}, Shape: {input_tensor.shape}, Type: {input_tensor.type}")

        This website can also be used to visualize onnx model and see inputs/outputs: https://netron.app/
        '''

        with open(model_path, 'rb') as model_file: #parse onnx file, write to network
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        builder.max_batch_size = 1 #max number of samples that infernece can be performed on at once
        #builder.max_workspace_size = 1 << 30 1Gb , when not defined default is total system memory

        profile= builder.create_optimization_profile()
        if self.use_yolov5:
            profile.set_shape("images", (1, 3, 416, 416), (1, 3, 416, 416), (1, 3, 416, 416))

        else:
            profile.set_shape("input_1", (1, 3, 416, 416), (1, 3, 416, 416), (1, 3, 416, 416)) #image input
            profile.set_shape("image_shape", (1, 2), (1, 2), (1, 2)) #original image shape input into yolo model

        config.add_optimization_profile(profile)

        half = rospy.get_param("~use_fp16")
        if half: #lessons precision to increase speed
            config.set_flag(trt.BuilderFlag.FP16)



        serialized_network = builder.build_serialized_network(network, config)

       
        with open(engine_path, "wb") as f:
            f.write(serialized_network)

        print(f"TensorRT engine saved to {engine_path}")

        

    def load_engine(self,engine_path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        

    def letterbox_image(self, image, size):
        #resize while maintaining aspect ratio using padding
        iw, ih = image.size
        w, h = size
        scale = min(w/float(iw), h/float(ih))
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image


    def letterbox_coords_to_img(self,image_shape,letterbox_shape, x,y):
        
        iw, ih = image_shape
        w, h = letterbox_shape

        # Calculate the scale and new dimensions after resizing
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        
        # Calculate the offsets
        offset_x = (w - nw) // 2
        offset_y = (h - nh) // 2

        # Reverse the scaling and offsets
        original_x = (x - offset_x) / scale
        original_y = (y - offset_y) / scale

        return int(original_x), int(original_y)


    def pre_process(self,img): #PIL image
        
        cv_img=self.bridge.imgmsg_to_cv2(img, desired_encoding='rgb8')
        #cv_img=cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image=Image.fromarray(cv_img) #will be used later for drawing bounding boxes

        model_image_size=(416,416)
        boxed_image = self.letterbox_image(self.pil_image, tuple(reversed(model_image_size)))
        
        image_data = np.array(boxed_image, dtype='float32')
        
        
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)

        image_size = np.array([image_rows, image_cols]).reshape(1, 2) #original input image size
        return image_data, image_size #numpy array image data, np array image size
         

       
        


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

            '''
            print(f"Binding Index: {binding_idx}")
            print(f"Binding Name: {binding_name}")
            print(f"Binding Shape: {binding_shape}")
            print(f"Data Type: {np_dtype}")
            print(f"Size (# elements): {size}")
            print(f"Binding Type: {binding_type}")
            print("=" * 40)

            '''
                     
            if binding_idx==4 and self.use_tiny_yolo==False and self.use_yolov5==False:
                size=3*self.max_nbox
            elif binding_idx==2 and self.use_tiny_yolo==True and self.use_yolov5==False:
                size=3*self.max_nbox
        
            host_mem=cuda.pagelocked_empty(size, np_dtype)
            device_mem=cuda.mem_alloc(host_mem.nbytes) 
            bindings.append(int(device_mem))

            

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem,device_mem, binding_name))
                   
            else:
                outputs.append(HostDeviceMem(host_mem,device_mem, binding_name))

            #Print info 
            
         

            self.inputs, self.outputs, self.bindings= inputs, outputs, bindings
                 

    def __del__(self):
        self.ctx.pop()
        del self.engine
        del self.context
        del self.runtime
       
        return

    def postprocess(self,scores, boxes, indices):
      
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_score=scores[tuple(idx_)]
            if out_score <self.min_score: 
                continue
            out_classes.append(idx_[1])
            out_scores.append(out_score)
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        
        return out_boxes, out_scores, out_classes
     
    def draw_boxes(self,src_image, boxes, class_index):
        #src image of type PIL image
        draw = ImageDraw.Draw(src_image)
        length= len(class_index)
        width=image_cols
        height=image_rows

        for i in range(length):
            y_min, x_min, y_max, x_max= boxes[i]

            class_pred= self.classes[class_index[i]]
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))
            x_av=(x_min+x_max)/2
            y_av=(y_min+y_max)/2

            draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red")
            # draw.rectangle((((x_av+x_min)/2, (y_av+y_min)/2), ((x_av+x_max)/2, (y_av+y_max)/2)), outline="blue")
            draw.rectangle((((45*x_max+55*x_min)/100, (45*y_max+55*y_min)/100), ((55*x_max+45*x_min)/100, (55*y_max+45*y_min)/100)), outline="blue")
            draw.text((x_max-(x_max-x_min)/2, y_max-(y_max-y_min)/2), class_pred)
            

        cv_result= cv2.cvtColor(np.array(src_image), cv2.COLOR_RGB2BGR)
        return cv_result


    def inference(self,input_image): #input image= sensormsgs/Image.msg

        self.input_image,self.image_shape=self.pre_process(input_image)

        self.input_image=np.ravel(self.input_image)
        self.image_shape=np.ravel(self.image_shape)
        

        
        
        self.ctx.push()
            
            
        #1)copy input data to input memory buffers,
        np.copyto(self.inputs[0].host, self.input_image)
        if self.use_yolov5==False:
            np.copyto(self.inputs[1].host,self.image_shape)

           
        #2)tranfer input data and bindings to GPU via cuda stream
        #buffer= memory accessible by gpu and cpu
        for inp in self.inputs: 
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        for inp in self.inputs:
            self.context.set_tensor_address(inp.name, int(inp.device))
               

        for out in self.outputs:
            self.context.set_tensor_address(out.name, int(out.device))

           
            
        #3)execute inference 
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
           
            
            #4)Transfer result from GPU (Device) to host (CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) #copy device data into host data for cpu usuage

            #use print statement in allocate buffers to obtain name and match model outputs 
            if self.use_yolov5:
                if out.name=='num_dets':
                    num_detections= np.copy(np.array(out.host)) #integer
                if out.name== 'boxes':
                    raw_boxes= np.copy(np.array(out.host))
                if out.name == 'scores':
                    raw_scores= np.copy(np.array(out.host))
                if out.name == 'labels':
                    raw_labels= np.copy(np.array(out.host))


            elif self.use_tiny_yolo==False:
                if out.name == 'yolonms_layer_1/ExpandDims_1:0':
                    raw_boxes= np.copy(np.array(out.host))
                    #print('boxes \n')
                elif out.name == 'yolonms_layer_1/ExpandDims_3:0':
                    #print('scores \n') 
                    raw_scores = np.copy(np.array(out.host))
                else:
                    raw_indices = np.copy(np.array(out.host))
                
            else:
                if out.name == 'yolonms_layer_1':
                    raw_boxes= np.copy(np.array(out.host))
                    #print('boxes \n')
                elif out.name == 'yolonms_layer_1:1':
                    #print('scores \n') 
                    raw_scores = np.copy(np.array(out.host))
                else:
                    raw_indices = np.copy(np.array(out.host))
            
            out.host[:]=0 #reset values for next inference call

            
            
        #5) synchronize cuda stream
            
        self.stream.synchronize()

        self.ctx.pop()

        #reshape results
        if self.use_yolov5:

            # Post Processing: 

            #rescale box location from 416x416 letterbox image format to 480x640 original image 

            #draw onto image (can likely reuse existing function self.draw_boxes())

            #create Yolo Message ( only class in custom model is the car )

            #return message and image for publishing 
            '''
            print(f"num_detections: {num_detections}")
            
            print(f"raw_boxes: {raw_boxes}")
            print(f"raw_scores: {raw_scores}")
            print(f"raw_labels: {raw_labels}")
            '''
            out_boxes =[]
            out_scores =[]
            out_classes =[]
            for i in range(num_detections[0]): #loop through each detection
                if raw_labels[i] < 0 or raw_scores[i] <0.4: #background object given value -1 according to documentation
                    continue
                
                start_box_idx=i*4
                xmin, ymin, xmax, ymax= raw_boxes[start_box_idx], raw_boxes[start_box_idx+1], raw_boxes[start_box_idx+2], raw_boxes[start_box_idx+3]
                #rescale
                '''
                print("before rescale \n")
                print(ymin, xmin , ymax, xmax)
                print('\n')
                '''
                xmin,ymin=self.letterbox_coords_to_img([640,480], [416,416], xmin,ymin)
                xmax,ymax=self.letterbox_coords_to_img([640,480], [416,416], xmax,ymax)
                
                out_boxes.append([ymin,xmin,ymax,xmax]) #ordering kept consistent with yolov3 box format
                
                out_scores.append(raw_scores[i])
                out_classes.append(raw_labels[i])
           #print("\n Out scores: ", out_scores)

            #create yolo_msg, draw and display result

            #print( out_boxes)
            boxes_array=[]
            for box in out_boxes:
                boxes_array.extend(box)
                
            yolo_msg=YoloData()
            yolo_msg.rectangles= boxes_array

            class_names=[]
            for idx in out_classes:
                class_names.append(self.classes[idx])
            
            yolo_msg.classes= class_names

            
            cv_result= self.draw_boxes(self.pil_image, out_boxes, out_classes)  
            image_message = self.bridge.cv2_to_imgmsg(cv_result, encoding="bgr8")

            return image_message, yolo_msg
                
                
      

     
        else:

            scores = raw_scores.reshape((1, 80, -1))
            boxes = raw_boxes.reshape((1, -1, 4))
            indices = np.unique(raw_indices.reshape((-1, 3)), axis=0) #reshape and remove duplicate boxes

            out_boxes, out_scores, out_classes= self.postprocess(scores,boxes,indices)
            boxes_array= [] #1D array where every 4 elements corresponds to 4 vertices of bounding box
            for box in out_boxes:
                y_min, x_min, y_max, x_max= box

                box[1] = max(0, min(box[1], image_cols))
                box[0] = max(0, min(box[0], image_rows))
                box[3] = max(0, min(box[3], image_cols))
                box[2] = max(0, min(box[2], image_rows))

                boxes_array.extend(box)  # Flattening the list
            

        

            
            
            class_names=[] #array of strings
            for index in out_classes:
                class_names.append(self.classes[index])

            yolo_msg= YoloData()
            yolo_msg.rectangles=  boxes_array
            yolo_msg.classes= class_names
            


            cv_result= self.draw_boxes(self.pil_image, out_boxes, out_classes)  
            image_message = self.bridge.cv2_to_imgmsg(cv_result, encoding="bgr8")

            return image_message, yolo_msg
                


class AEV_Detector():
    def __init__(self):
        rgb_image_topic=rospy.get_param("~rgb_image_topic")
        YOLO_topic= rospy.get_param("~yolo_topic")
        data_topic= rospy.get_param("~yolo_data_topic")
        
        self.model=VisionModel()
        

        rospy.Subscriber(rgb_image_topic,ROSImage,self.rgb_callback, queue_size=1)

        self.yolo_pub= rospy.Publisher(YOLO_topic, ROSImage, queue_size=1)
        self.data_pub= rospy.Publisher(data_topic, YoloData,queue_size=1)
        


    def rgb_callback(self, data):
        
            result, out_data =self.model.inference(data)
            self.yolo_pub.publish(result)
            self.data_pub.publish(out_data) 
         
        

        



def main(args):
    rospy.init_node("YOLO", anonymous=True)
    vehicles=AEV_Detector()
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
