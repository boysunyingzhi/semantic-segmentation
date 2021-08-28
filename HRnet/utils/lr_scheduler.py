
def lr_poly(base_lr,i_iter,max_iter,power):
    return base_lr*((1-float(i_iter)/max_iter)**(power))

def adjust_learning_rate_poly(init_lr,optimizer,i_iter,max_iter):
    lr = lr_poly(init_lr,i_iter,max_iter,0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

