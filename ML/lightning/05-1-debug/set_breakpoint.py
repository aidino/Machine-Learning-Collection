def function_to_debug():
    x = 2

    # set breakpoint
    import pdb
    pdb.set_trace()
    y = x**2
    
    
if __name__ == '__main__':
    function_to_debug()