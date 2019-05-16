from SpatialMux_continue import *

if __name__ == '__main__':
    mux = SpatialMux(1.55)
    am = mux.am
    am1 = mux.am1
    am2 = mux.am2
    
    sim1 = mux.sim1
    sim2 = mux.sim2
    params = mux.params0
    field_monitor = mux.field_monitor
    
    print len(params)
    if NOT_PARALLEL:
        roc_fom, x_store, y_store, ids_store = am1.calc_roc_fom(params)
        
        xxx = np.array([])
        yyy = np.array([])


        for i in range(len(x_store)):
            for j in range(len(x_store[i])):
                xxx = np.append(xxx, x_store[i][j])
                yyy = np.append(yyy, y_store[i][j])
        
        import matplotlib.pyplot as plt
        
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(xxx,yyy, '-o')
        plt.axis('equal')
        
        plt.show()
