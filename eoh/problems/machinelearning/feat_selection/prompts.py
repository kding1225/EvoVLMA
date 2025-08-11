import json

class GetPrompts():
    def __init__(self):
        
        self.prompt_task = "Given CLIP's encoded features of train images and class names, you need to select the best feature channels. Your selected channels should improve classification accuracy on test images. Help me design a novel feature selection algorithm that is different from the ones in literature."
        self.prompt_func_name = 'feat_selection'
        self.prompt_func_inputs = ['clip_weights', 'train_feats', 'w0', 'w1', 'topk']
        self.prompt_func_outputs = ['indices']
        self.prompt_inout_inf = "'clip_weights' denotes the zero-shot classifier weights computed by encoding all the class names by CLIP's text encoder, which has a shape of (c,d) with c the number of classes and d the number of feature dimensions. Each row of 'clip_weights' is L2-normalized. 'train_feats' denotes the train images' features, which is a matrix of shape (c,k,d) with k the number of train images per class. The third dimension of 'train_features' is L2-normalized. 'w0' and 'w1' are preset hyper-parameters, which are in the range [0,1]. 'topk' denotes the number of selected features. 'indices' denotes the indices of selected feature channels, which should have a length of 'topk'."
        self.prompt_other_inf = "All except 'w0', 'w1', 'topk' are PyTorch Tensors. Do not introduce randomness in the code. Do not introduce any learnable parameters. Please optimize runtime efficiency while maintaining code readability, avoiding the use of deep nested loops. You can use any mathematical operation on the inputs, please try to be creative and make full use of the input information."
        
    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
