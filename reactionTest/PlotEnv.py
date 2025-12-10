import matplotlib.pyplot as plt
import matplotlib
import scienceplots


class PlotEnv:

    def __init__(
        self,
        dpi=120,
        format="pdf",
        lwc=0.5,
        msc=5,
        markEvery=400,
        font_options={"size": 12},
    ):
        plt.style.use(["science"])
        matplotlib.rc("font", **font_options)
        self.dpi = dpi
        self.format = format
        self.lwc = lwc
        self.msc = msc
        self.markEvery = markEvery
        self.markerList = (
            ".",
            "s",
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
        )
        self.color_seq = matplotlib.color_sequences["tab10"]

    def figure(self, num=None, figsize=(6, 4)):
        return plt.figure(num, figsize=figsize, dpi=self.dpi)

    def plot(
        self,
        *args,
        plotIndex=0,
        **kwargs,
    ):

        plt.plot(
            *args,
            **kwargs,
            lw=self.lwc,
            marker=self.markerList[plotIndex % len(self.markerList)] if self.markEvery > 0 else None,
            markevery=self.markEvery,
            markersize=self.msc,
            markeredgewidth=self.lwc,
            markerfacecolor="none",
            color=self.color_seq[plotIndex % len(self.color_seq)],
        )
