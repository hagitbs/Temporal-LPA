from copy import deepcopy
from typing import List, Literal
import numpy as np
import altair as alt
import pandas as pd


def metric_bar_chart(df, rule_value, metric):
    df = deepcopy(df)
    df["color"] = np.where(df[metric] < rule_value, True, False)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:O", axis=alt.Axis(labels=False)),
            y=alt.Y(metric),  # , scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("color", legend=None),
        )
    ).properties(height=250)
    df["y"] = rule_value
    rule = alt.Chart(df).mark_rule(color="red").encode(y="y")
    return chart + rule


def timeline(
    df,
    x,
    y,
    title,
    stack: Literal["stack", "center"] | None,
    order: List[str] | None = None,
    name="timeline",
) -> alt.Chart:
    """
    Creates colorful timeline
    Parameters
    ----------
    stack : Literal["stack", "center"] | None
        "stack" for a timeline for zero, including minus values (good for showing distances for instance)
        "center" for a timeline centered aroud the middle of the y axis (good for showing frequencies)
    order : List[str]
        a list of the values in the wanted order (for instance ordering elements with  distance from highest to lowest)
    """
    selection = alt.selection_multi(fields=["element"], bind="legend")
    xx = df[x].drop_duplicates().to_list()
    values = xx[::5] if len(xx) > 100 else xx
    if not order:
        order = df["element"].drop_duplicates().to_list()
    if len(order) > 1:
        color = alt.Color(
            "element",
            scale=alt.Scale(scheme="rainbow"),
            sort=alt.Sort(order),
            legend=alt.Legend(columns=2, labelLimit=1000),
        )
    elif len(order) == 1:
        color = alt.value("blue")
    chart = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X(f"{x}:O", title="Date", axis=alt.Axis(values=values)),
            y=alt.Y(
                f"{y}",
                stack=stack,
                title=f"{y.replace('_', ' ').capitalize()}",
            ),
            color=color,
            tooltip=alt.Tooltip(["element", y]),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .interactive()
        .properties(width=900)
        .add_selection(selection)
    )
    # chart.save(f"results/{corpus}/timelines/{name}.html")
    return chart


def facet_timeline(
    df,
    x,
    y,
    title,
    stack: Literal["stack", "center"] | None,
    order: List[str] | None = None,
    name="timeline",
    color=None,
    width: int = 900,
) -> alt.Chart:
    """
    Creates colorful timeline
    Parameters
    ----------
    stack : Literal["stack", "center"] | None
        "stack" for a timeline for zero, including minus values (good for showing distances for instance)
        "center" for a timeline centered aroud the middle of the y axis (good for showing frequencies)
    order : List[str]
        a list of the values in the wanted order (for instance ordering elements with  distance from highest to lowest)
    """
    values = (
        list(range(1790, 2022, 5))
        if len(df[x].drop_duplicates()) > 230
        else df[x].tolist()
    )
    if not order:
        order = df["element"].drop_duplicates().to_list()
    if len(order) > 1:
        color = alt.Color(
            "element",
            scale=alt.Scale(scheme="rainbow"),
            sort=alt.Sort(order),
            legend=alt.Legend(columns=2, labelLimit=1000),
        )
        height = {}
    elif len(order) == 1:
        color = alt.value(color)
        height = {"height": 100}
    chart = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X(f"{x}:O", title="Date", axis=alt.Axis(values=values)),
            y=alt.Y(f"{y}", stack=stack, axis=alt.Axis(title=None)),
            color=color,
            tooltip=alt.Tooltip(["element", y]),
        )
        .properties(width=width, **height)
    )
    # chart.save(f"results/{corpus}/timelines/{name}.html")
    return chart


def moving_avg(df):
    base = alt.Chart(df).encode(alt.X("index", axis=alt.Axis(title=None)))
    line = base.mark_line().encode(
        alt.Y("ma:Q", scale=alt.Scale(type="log"))
        # axis=alt.Axis(title='Precipitation (inches)', titleColor='#5276A7')
    )
    area = base.mark_area(opacity=0.3, color="#57A44C").encode(
        alt.Y("max:Q", scale=alt.Scale(type="log")),
        #   axis=alt.Axis(title='Avg. Temperature (°C)', titleColor='#57A44C')),
        alt.Y2("min:Q"),
    )
    return alt.layer(area, line)


def sockpuppet_matrix(
    spd, colorscheme="yellowgreenblue", date: bool = False
) -> alt.Chart:
    c1n, c2n = (c for c in spd.columns if c != "value")
    values = (
        spd[c1n].drop_duplicates().tolist()[::5]
        if date
        else spd[c1n].drop_duplicates().tolist()
    )
    return (
        alt.Chart(spd)
        .mark_rect()
        .encode(
            x=alt.X(f"{c1n}:O", axis=alt.Axis(orient="top", values=values), sort=None),
            y=alt.Y(
                f"{c2n}:O", axis=alt.Axis(orient="right", values=values), sort=None
            ),
            color=alt.Color("value", scale=alt.Scale(scheme=colorscheme)),
        )
        .configure_axis(title=None)
        .configure_view(strokeWidth=0)
    )


def plot_pca(data, labels):
    data = pd.DataFrame(data, columns=["x", "y"])
    data["label"] = labels
    chart = alt.Chart(data).mark_circle().encode(x="x", y="y", tooltip=["label"])
    text = chart.mark_text(align="left", baseline="middle", dx=7).encode(text="label")
    return chart + text
