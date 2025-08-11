
class GetPrompts():
    def __init__(self):

        self.prompt_task = "You are given a task for adapting CLIP for few-shot image classification. You need to devise a good algorithm for computing the classification logits of test images. By finding the index corresponing to the maximal item in the logits, we can obtain the predictive labels. The devised algorithm should be different from those in the existing literature."
        self.prompt_func_name = "compute_logits"
        self.prompt_func_inputs = ['train_feats', 'train_labels', 'test_feats', 'clip_weights', 'w0', 'w1', 'topk', 'alpha0', 'alpha1', 'alpha2']
        self.prompt_func_outputs = ["logits"]
        self.prompt_inout_inf = "'train_feats' denotes the train images' features, which is a matrix of shape (c,k,d) with c the number of classes, k the number of train images per class, d the number of feature dimensions. The third dimension of 'train_feats' is L2-normalized. 'train_labels' denotes the train labels with a shape of (c,k). The values of the i-th row in 'train_labels' are i. 'test_feats' denotes the test images' features of shape (n,d), where n is the number of test samples. 'clip_weights' denotes the zero-shot classifier weights computed by encoding all the class names by CLIP's text encoder, which has a shape of (c,d). Each row of 'clip_weights' is L2-normalized. 'w0', 'w1' and 'topk' are preset hyper-parameters for selecting important feature channels. 'w0' and 'w1' are used for defining criterion for feature selection, which are in the range [0,1], 'topk' denotes the number of selected channels. 'alpha0', 'alpha1', 'alpha2' are optional hyper-parameters in logits function, they are in the range [0, 20]. You could use some or all of these hyper-parameters."
        self.prompt_other_inf = "You only need to design the feature selection function 'feat_selection' and the logits computation function 'compute_logits_with_fs' using selected features. Do not modify the input parameter names of these two functions. All except 'w0', 'w1', 'topk', 'alpha0', 'alpha1' and 'alpha2' are PyTorch Tensors. Do not introduce randomness in the code. Do not introduce any learnable parameters. Please optimize runtime efficiency while maintaining code readability, avoiding the use of deep nested loops. You can use any mathematical operation on the inputs, please try to be creative and make full use of the input information."
        
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
