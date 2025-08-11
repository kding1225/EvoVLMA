
class GetPrompts():
    def __init__(self, use_fs):

        if use_fs:
            self.prompt_task = "You are given a task for adapting CLIP for few-shot image classification. You need to devise a good function for computing the classification logits of test images. Actually, you need to combine the information in text features, test features, train features and train labels. Also, the set of important feature channels is provided. You should consider to use this information in the devised function. By finding the index corresponing to the maximal item in the logits outputted by this function, we can obtain the predictive labels. The devised logits function should be different from those in the existing literature."
        else:
            self.prompt_task = "You are given a task for adapting CLIP for few-shot image classification. You need to devise a good function for computing the classification logits of test images. Actually, you need to combine the information in text features, test features, train features and train labels. By finding the index corresponing to the maximal item in the logits outputted by this function, we can obtain the predictive labels. The devised logits function should be different from those in the existing literature."
        
        self.prompt_func_name = "compute_logits"
        if use_fs:
            self.prompt_func_inputs = ['train_feats', 'train_labels', 'test_feats', 'clip_weights', 'indices', 'alpha0', 'alpha1', 'alpha2']
        else:
            self.prompt_func_inputs = ['train_feats', 'train_labels', 'test_feats', 'clip_weights', 'alpha0', 'alpha1', 'alpha2']
    
        self.prompt_func_outputs = ["logits"]
        if use_fs:
            self.prompt_inout_inf = "'train_feats' denotes the train images' features, which is a matrix of shape (c,k,d) with c the number of classes, k the number of train images per class, d the number of feature dimensions. The third dimension of 'train_feats' is L2-normalized. 'train_labels' denotes the train labels with a shape of (c,k). The values of the i-th row in 'train_labels' are i. 'test_feats' denotes the test images' features of shape (n,d), where n is the number of test samples. 'clip_weights' denotes the zero-shot classifier weights computed by encoding all the class names by CLIP's text encoder, which has a shape of (c,d). Each row of 'clip_weights' is L2-normalized. 'indices' denotes the indices of important features. 'alpha0', 'alpha1', 'alpha2' are optional hyper-parameters, they are in the range [0, 20]. You could use some or all of these hyper-parameters."
        else:
            self.prompt_inout_inf = "'train_feats' denotes the train images' features, which is a matrix of shape (c,k,d) with c the number of classes, k the number of train images per class, d the number of feature dimensions. The third dimension of 'train_feats' is L2-normalized. 'train_labels' denotes the train labels with a shape of (c,k). The values of the i-th row in 'train_labels' are i. 'test_feats' denotes the test images' features of shape (n,d), where n is the number of test samples. 'clip_weights' denotes the zero-shot classifier weights computed by encoding all the class names by CLIP's text encoder, which has a shape of (c,d). Each row of 'clip_weights' is L2-normalized. 'alpha0', 'alpha1', 'alpha2' are optional hyper-parameters, they are in the range [0, 20]. You could use some or all of these hyper-parameters."
        self.prompt_other_inf = "All except 'alpha0', 'alpha1' and 'alpha2' are PyTorch Tensors. Do not introduce randomness in the code. Do not introduce any learnable parameters. Please optimize runtime efficiency while maintaining code readability, avoiding the use of deep nested loops. You can use any mathematical operation on the inputs, please try to be creative and make full use of the input information."
        # In code, each parameter can appear at most once.
        
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
