import numpy as np

def balanced_subsample(x,y,subsample_size=1.0,possible_y=[1,0]):
    y = np.array(y)
    class_xs = []
    min_elems = None

    for yi in possible_y:
        elems = x[np.where(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    else :
        use_elems = int(min_elems*subsample_size)
    
    
    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(len(x_))
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

if __name__ == '__main__':
    x = np.array([[11,21]]*100 + [[10,20]]*10);
    y = np.array([1]*100 + [0]*10);
    
    bal_set = balanced_subsample(x,y,subsample_size=2)
    print len(bal_set[0]),len(bal_set[1]), bal_set

