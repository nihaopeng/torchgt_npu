from general import parallel_bar

if __name__ == "__main__":
    params = {
        'font.family': 'Arial',
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 16,
        'figsize':(10,4)
    }
    
    # 配置参数：通过 use_twin 开关控制
    my_params = {
        'axes': [1, 1],
        'bar_width': 0.5,
        'legend_loc':'lower center',
        'bbox_to_anchor':(0.5, 0.88),
        'ncol':3,
        'gap':2,# 调整横轴组之间的间距
        'axes_params' : [
            {
                'y_val' : {
                    'Baseline': [1, 1, 1, 1],
                    'Baseline+HAW': [1.670451059, 35.75071297, 14.71996592, 11.42793054],
                    'Baseline+HAW+IWP' : [16.28964692,313.6356255,295.6288243,59.89626556],
                },
                'x_ticks' : ['OAV','AZ','OPT','RDT'],
                'title' : '',
                'y1_lim' : (0.01, 1000),
                # 'y2_lim' : (1e-5, 10),
                'y1_log' : True,
                # 'y2_log' : True,
                'y1_label' : 'Speed Up',
                # 'y2_label' : 'Capture Rate',
                'hatchs' : ['///', '...', '\\\\\\'],
                'colors' : ['#1f77b4', '#ff7f0e', '#ffff0e'],
                # 'use_twin': True,              # 是否开启双 Y 轴
            }
        ]
    }

    parallel_bar(params, my_params, figpath='ablation.pdf')
