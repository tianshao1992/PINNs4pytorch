import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import matplotlib.cm as cm


class matplotlib_vision(object):

    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name
        # self._cbs = [None] * len(self.field_name) * 3

        # gs = gridspec.GridSpec(1, 1)
        # gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        # gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        # self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}


    def plot_loss(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper left", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_value(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper left", prop=self.font)
        plt.xlabel('variable', self.font)
        plt.ylabel('pred value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_scatter(self, true, pred, axis=0, title=None):
        # sbn.set(color_codes=True)

        plt.scatter(np.arange(true.shape[0]), true, marker='*')
        plt.scatter(np.arange(true.shape[0]), pred, marker='.')

        plt.ylabel('target value', self.font)
        plt.xlabel('samples', self.font)
        plt.xticks(fontproperties='Times New Roman', size=25)
        plt.yticks(fontproperties='Times New Roman', size=25)
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)

    def plot_regression(self, true, pred, axis=0, title=None):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        plt.scatter(true, pred, marker='.')

        plt.plot([min_value, max_value], [min_value, max_value], 'r-', linewidth=5.0)
        plt.fill_between([min_value, max_value], [0.95 * min_value, 0.95 * max_value],
                         [1.05 * min_value, 1.05 * max_value],
                         alpha=0.2, color='b')

        # plt.ylim((min_value, max_value))
        plt.xlim((min_value, max_value))
        plt.ylabel('pred value', self.font)
        plt.xlabel('real value', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)
        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)

    def plot_error(self, error, title=None):
        # sbn.set_color_codes()
        error = pd.DataFrame(error) * 100
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "g"}, fit_kws={"color": "r", "lw": 3}, hist_kws={"color": "b"})
        # plt.xlim([-1, 1])
        plt.xlabel("predicted relative error / %", self.font)
        plt.ylabel('distribution density', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)
        # plt.legend()
        plt.title(title, self.font)

    def plot_fields1d(self, curve, true, pred, name):
        plt.plot(curve, true)
        plt.plot(curve, pred)
        plt.xlabel('x coordinate', self.font)
        plt.ylabel(name, self.font)
        plt.yticks(fontproperties='Times New Roman', size=15)
        plt.xticks(fontproperties='Times New Roman', size=15)

    def plot_fields_tri(self, out_true, out_pred, coord, cell, cmin_max=None, fmin_max=None, field_name=None,
                        cmap='RdBu_r', ):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0,)), out_true.max(axis=(0,))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, 0]
        y_pos = coord[:, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 15
            triObj = tri.Triangulation(x_pos, y_pos, triangles=cell)  # 生成指定拓扑结构的三角形剖分.

            Num_levels = 20
            # plt.triplot(triObj, lw=0.5, color='white')

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########      Exact f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3 * fi + 1)
            levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_true[:, fi], Num_levels, cmap=cmap)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 15
            plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('True field $' + field_name[fi] + '$' + '', fontsize=15)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Learned f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3 * fi + 2)
            # levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_pred[:, fi], Num_levels, cmap=cmap)
            cb = plt.colorbar()
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 15
            plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Pred field $' + field_name[fi] + '$' + '', fontsize=15)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Error f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3 * fi + 3)
            err = out_pred[:, fi] - out_true[:, fi]
            plt.tricontourf(triObj, err, Num_levels, cmap='coolwarm')
            cb = plt.colorbar()
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 15
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error$' + field_name[fi] + '$' + '', fontsize=15)

    def plot_fields_ms(self, out_true, out_pred, coord, cmin_max=None, fmin_max=None, field_name=None,
                       cmap='RdBu_r', ):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0, 1)), out_true.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 15

            ########      Exact f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3*fi+1)
            f_true = out_true[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_true, cmap=cmap, shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, f_true, cmap='jet',)
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 15
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('True field $' + field_name[fi] + '$' + '', fontsize=15)


            ########     Learned f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3*fi+2)
            f_pred = out_pred[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_pred, cmap=cmap, shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, f_pred, cmap='jet',)
            cb = plt.colorbar()
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 15
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Pred field $' + field_name[fi] + '$' + '', fontsize=15)

            ########     Error f(t,x,y)     ###########
            plt.subplot(Num_fields, 3, 3*fi+3)
            err = f_true - f_pred
            plt.pcolormesh(x_pos, y_pos, err, cmap='coolwarm', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, err, cmap='coolwarm', )
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            cb = plt.colorbar()
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 15
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error$' + field_name[fi] + '$' + '', fontsize=15)

    def plot_fields_am(self, out_true, out_pred, coord, p_id, fig):

        fmax = out_true.max(axis=(0, 1, 2))  # 云图标尺
        fmin = out_true.min(axis=(0, 1, 2))  # 云图标尺

        def anim_update(t_id):
            print('para:   ' + str(p_id) + ',   time:   ' + str(t_id))
            axes = self.plot_fields_ms(out_true[t_id], out_pred[t_id], coord[t_id], fmin_max=(fmin, fmax))
            return axes

        anim = FuncAnimation(fig, anim_update,
                             frames=np.arange(0, out_true.shape[0]).astype(np.int64), interval=200)

        anim.save(self.log_dir + "\\" + str(p_id) + ".gif", writer='pillow', dpi=300)
