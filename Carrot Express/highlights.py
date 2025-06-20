import pandas as pd
import plotly.express as px
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import numpy as np

print(np.__version__)
data = pd.read_csv("./data/april_28_new.csv", parse_dates=["WeekStartDate"])

# Store 963290 (West Kendall (London Square))
# Store 963260 (Weston)
# Store 963280 (Pinecrest)


def preprocess(items):
    max_date = items["WeekStartDate"].max()

    # Drop rows with the max date
    items = items[items["WeekStartDate"] != max_date]

    print(items.groupby("StoreId")["WeekStartDate"].agg(["min", "max"]).reset_index())

    remove_ordertypes = [
        "SimpleCater",
        "EZ Cater (Pickup)",
        "Catering (Pickup)",
        "EZ Cater (Delivery)",
        "Olo Catering (Self-Delivery))",
        "Olo Catering (Pickup)",
    ]
    items = items[~items["OrderType"].isin(remove_ordertypes)]
    items.OrderType.unique()

    # remove items starting with google order as they  are not clear, can't get info from their item id, same ids are used for differtn items
    items1 = items[~items["MenuItemName"].str.contains("Google Order")]
    items1["MenuItemName"].nunique()

    items1["OrderType"] = items1["OrderType"].replace(
        {
            "Uber Eats - Delivery!": "UEats-Delivery",
            "Online Ordering (Pickup) *": "Online-Pickup",
            "UberEats (Pickup)": "UEats-Pickup",
            "Online Ordering (Dispatch) *": "Online-Dispatch",
            "Telephone - Pickup": "Telephone-Pickup",
            "Dine In, Dine In": "Dine In",
            "Take Out, Take Out": "Take Out",
        }
    )

    print("number of unique_item_names", items1["MenuItemName"].nunique())
    print(items1["OrderType"].value_counts())

    items1["item_weekly_revenue"] = items1["item_weekly_revenue"].astype(float)
    return items1


preprocessed_data = preprocess(data)


def viz(result):
    result = result.copy()

    for i in [963280, 963260, 963290]:
        df = result[result["StoreId"] == i]

        # Ensure WeekStartDate is datetime
        df["WeekStartDate"] = pd.to_datetime(df["WeekStartDate"])

        # Get unique StoreIds and OrderTypes for subplot grid
        store_ids = df["StoreId"].unique().tolist()
        order_types = df["OrderType"].unique().tolist()

        n_rows, n_cols = len(store_ids), len(order_types)

        # Create subplot grid
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=[
                f"{store} — {order}" for store in store_ids for order in order_types
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.07,
        )

        # Keep track of shown legend items for MenuItemName
        shown_items = set()

        for i, store in enumerate(store_ids, start=1):
            for j, order in enumerate(order_types, start=1):
                sub_df = df[(df["StoreId"] == store) & (df["OrderType"] == order)]
                if sub_df.empty:
                    continue

                for item in sub_df["MenuItemName"].unique():
                    item_df = sub_df[sub_df["MenuItemName"] == item].sort_values(
                        "WeekStartDate"
                    )
                    x = item_df["WeekStartDate"]
                    y = item_df["item_weekly_volumn"]

                    if len(item_df) < 2:
                        continue  # skip if not enough points

                    # Show legend only once per MenuItemName globally
                    show_leg = item not in shown_items
                    shown_items.add(item)

                    # 1) Plot actual weekly volume line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines+markers",
                            name=item,
                            showlegend=show_leg,
                            legendgroup=item,
                            hoverinfo="name+x+y",
                        ),
                        row=i,
                        col=j,
                    )

                    # 2) Compute linear regression for trend line
                    x_ord = x.map(pd.Timestamp.toordinal).values
                    slope, intercept = np.polyfit(x_ord, y, 1)
                    y_hat = slope * x_ord + intercept

                    # Calculate R^2
                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    # 3) Plot trend line (dashed)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_hat,
                            mode="lines",
                            line=dict(dash="dash", color="green"),
                            showlegend=False,
                            hoverinfo="none",
                        ),
                        row=i,
                        col=j,
                    )

                    # 4) Add slope and R^2 annotation (top left corner of subplot)
                    sp_idx = (i - 1) * n_cols + j
                    xref = f"x{sp_idx}" if sp_idx > 1 else "x"
                    yref = f"y{sp_idx}" if sp_idx > 1 else "y"

                    fig.add_annotation(
                        xref=xref,
                        yref=yref,
                        x=min(x),
                        y=max(y_hat),
                        # text=f"{item}<br>slope={slope:.2f}<br>R²={r2:.2f}",
                        #  text=f"{item}:{slope:.2f}",
                        showarrow=False,
                        font=dict(size=9, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        xanchor="left",
                        yanchor="top",
                    )

        # Final layout tweaks
        fig.update_yaxes(matches=None)
        fig.update_layout(
            height=400 * n_rows,
            width=500 * n_cols,
            title="Weekly Item Volume with Trend Lines & Score Value by StoreId and OrderType",
            legend_title_text="MenuItemName",
            hovermode="x unified",
        )

        fig.show()

def high_revenue_items(preprocessed_data):
     # Aggregate volume share per MenuItemName across all weeks and stores
    item_volume_share = (
        preprocessed_data.groupby("StoreId")["item_weekly_revenue"].sum().reset_index()
    )
    item_volume_share.rename(
        columns={"item_weekly_revenue": "store_revenue_share"}, inplace=True
    )
    item_volume = (
        preprocessed_data.groupby(["StoreId", "MenuItemName"])["item_weekly_revenue"]
        .sum()
        .reset_index()
    )

    high = item_volume.merge(
        item_volume_share, on="StoreId", how="left", validate="m:1"
    )

    high["revenue_share"] = high["item_weekly_revenue"] / high["store_revenue_share"]

    check = high.groupby("StoreId")["revenue_share"].sum()
    print(check.head())

    # Step 1: For each store, compute the 75th percentile
    high["store_75th"] = high.groupby("StoreId")["revenue_share"].transform(
        lambda x: x.quantile(0.75)
    )

    # Step 2: Filter items that are >= 75th percentile *within their store*
    high_items = high[high["revenue_share"] >= high["store_75th"]]

    high_items.MenuItemName.nunique()

    high_volume_items = preprocessed_data[
        preprocessed_data["MenuItemName"].isin(high_items["MenuItemName"])
    ]
    print(
        "number of items in high volume share items",
        high_volume_items.MenuItemName.nunique(),
    )

    # looking for top 1o items in each store
    top10_items_per_store = (
        high.sort_values(["StoreId", "revenue_share"], ascending=[True, False])
        .groupby("StoreId")
        .head(10)
    )

    # Optional: Reset index for clean output
    top10_items_per_store.reset_index(drop=True, inplace=True)

    # Result

    # # Get top 30 items
    top_items_all = top10_items_per_store.sort_values(
        by="revenue_share", ascending=False
    )
    top_items_all['revenue_share']=top_items_all['revenue_share']*100

  
    # for i in top_items_all.StoreId.unique():
        
    #     fig = px.bar(
    #         top_items_all[top_items_all['StoreId']==i],
    #         x="revenue_share",
    #         y="MenuItemName",
    #         orientation="h",
    #         facet_col="StoreId",
    #         title="Top 10 Menu Items by Total Revenue Share",
    #         labels={"revenue_share": "Revenue Share", "MenuItemName": "Menu Item"},
    #         color="revenue_share",
    #         color_continuous_scale="temps",
    #         width=600,
    #         text="revenue_share",  # Add text labels
    #     )

    #     # Reverse y-axis so top item is at the top
    #     fig.update_layout(yaxis=dict(autorange="reversed"))
    #        # Reverse y-axis so top item is at the top
    #     fig.update_layout(
    #         yaxis=dict(autorange="reversed"),
    #         font=dict(size=10),  # Controls all base text size (axes labels, tick labels, legend)
    #         title=dict(font=dict(size=14)),  # Optional: slightly larger title
    #     )
        
 
    # # Format text on bars (e.g., as percentage with 1 decimal)
    #       # Format and resize text labels
    #     fig.update_traces(
    #         texttemplate='%{text:.1f}%',
    #         textposition='inside',
    #         textfont_size=10 # Set text size here
    #     )

    #     fig.show()
        
        
   
    fig = px.bar(
        top_items_all,
        x="revenue_share",
        y="MenuItemName",
        orientation="h",
        facet_col="StoreId",
        title="Top 10 Menu Items by Total Revenue Share",
        labels={"revenue_share": "Revenue Share", "MenuItemName": "Menu Item"},
        color="revenue_share",
        color_continuous_scale="temps",
        text="revenue_share",
        
    )

    # Reverse y-axis so top item is at the top
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='inside',
            textfont_size=10 # Set text size here
        )
    fig.update_layout(
            yaxis=dict(autorange="reversed"),
            font=dict(size=10),  # Controls all base text size (axes labels, tick labels, legend)
            title=dict(font=dict(size=14)),  # Optional: slightly larger title
        )
        
 

    fig.show()
    


    return high_volume_items  

    
# Legal Wrap, Chicken Caesar Wrap,Poke Bowl remains top 3 item per revenue share in all three stores.
# Liv wrap with position 4 in Pinecrest and West Kendall but position 5 in Weston.

    

    

high_revenue_items = high_revenue_items(preprocessed_data)


def get_hig_moving_items(preprocessed_data):
    # Aggregate volume share per MenuItemName across all weeks and stores
    item_volume_share = (
        preprocessed_data.groupby("StoreId")["item_weekly_volumn"].sum().reset_index()
    )
    item_volume_share.rename(
        columns={"item_weekly_volumn": "store_volume_share"}, inplace=True
    )
    item_volume = (
        preprocessed_data.groupby(["StoreId", "MenuItemName"])["item_weekly_volumn"]
        .sum()
        .reset_index()
    )

    high = item_volume.merge(
        item_volume_share, on="StoreId", how="left", validate="m:1"
    )

    high["volume_share"] = high["item_weekly_volumn"] / high["store_volume_share"]

    check = high.groupby("StoreId")["volume_share"].sum()
    print(check.head())

    # Step 1: For each store, compute the 75th percentile
    high["store_75th"] = high.groupby("StoreId")["volume_share"].transform(
        lambda x: x.quantile(0.75)
    )

    # Step 2: Filter items that are >= 75th percentile *within their store*
    high_items = high[high["volume_share"] >= high["store_75th"]]

    high_items.MenuItemName.nunique()

    high_volume_items = preprocessed_data[
        preprocessed_data["MenuItemName"].isin(high_items["MenuItemName"])
    ]
    print(
        "number of items in high volume share items",
        high_volume_items.MenuItemName.nunique(),
    )

    # looking for top 1o items in each store
    top10_items_per_store = (
        high.sort_values(["StoreId", "volume_share"], ascending=[True, False])
        .groupby("StoreId")
        .head(10)
    )

    # Optional: Reset index for clean output
    top10_items_per_store.reset_index(drop=True, inplace=True)

    # Result

    # # Get top 30 items
    top_items_all = top10_items_per_store.sort_values(
        by="volume_share", ascending=False
    )
    top_items_all['volume_share']=top_items_all['volume_share']*100
    fig = px.bar(
        top_items_all,
        x="volume_share",
        y="MenuItemName",
        orientation="h",
        facet_col="StoreId",
        title="Top 30 Menu Items by Total Volume Share",
        labels={"volume_share": "Total Volume Share", "MenuItemName": "Menu Item"},
        color="volume_share",
        color_continuous_scale="temps",
    )

    # Reverse y-axis so top item is at the top
    fig.update_layout(yaxis=dict(autorange="reversed"))

    fig.show()
    
    

    return high_volume_items

high_volume_items = get_hig_moving_items(preprocessed_data)


def declining_items(high_volume_items,var):
    # Prepare a results list
    results = []

    # Group by StoreId and OrderType
    for (store_id, order_type), group in high_volume_items.groupby(
        ["StoreId", "OrderType"]
    ):
        trend_data = []

        # Further group by MenuItemName
        for menu_item, item_group in group.groupby("MenuItemName"):
            item_group = item_group.sort_values("WeekStartDate")

            # Encode time as numeric (e.g., week number)
            item_group["week_num"] = (
                item_group["WeekStartDate"] - item_group["WeekStartDate"].min()
            ).dt.days

            # Need at least 4 data points to compute a meaningful trend
            if len(item_group) >= 3:
                X = item_group["week_num"].values.reshape(-1, 1)
                y = item_group["item_weekly_volumn"].values
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                # print(f"Menu Item: {menu_item}, Slope: {slope}")

                # Append slope
                trend_data.append((menu_item, slope))

        # Sort by most negative slope (strongest downward trend)
        trend_data.sort(key=lambda x: x[1])

        # Take top 5
        top_5_negative = trend_data[:5]

        for menu_item, slope in top_5_negative:
            results.append(
                {
                    "StoreId": store_id,
                    "OrderType": order_type,
                    "MenuItemName": menu_item,
                    "TrendSlope": slope,
                }
            )

    # Convert results to DataFrame
    top_declining_items = pd.DataFrame(results)

    print(
        "number of unique declining items",
        top_declining_items["MenuItemName"].nunique(),
    )

    df_tren = high_volume_items.merge(
        top_declining_items, on=["StoreId", "OrderType", "MenuItemName"]
    )
    df_tren.head(5)
    
    # Step 1: Filter only items with negative TrendSlope
    neg_trend = df_tren[df_tren["TrendSlope"] < 0]

    # # Step 2: Count how many stores each MenuItemName appears in (with a negative slope)
    # decline_counts = (
    #     neg_trend.groupby(["OrderType", "MenuItemName"])["StoreId"]
    #     .nunique()
    #     .reset_index(name="DecliningStoreCount")
    # )

    # # Step 3: Total number of stores
    # total_stores = high_volume_items["StoreId"].nunique()

    # # Step 4: Keep only those items that declined in *all* stores
    # items_declined_everywhere = decline_counts[
    #     decline_counts["DecliningStoreCount"] == total_stores
    # ]

    # # Step 5: Merge back to get full info
    # result = neg_trend.merge(
    #     items_declined_everywhere[["OrderType", "MenuItemName"]],
    #     on=["OrderType", "MenuItemName"],
    #     how="inner",
    # )

    viz(neg_trend)

    return result

result = declining_items(high_volume_items,var="item_weekly_volumn")


result[
    (result["MenuItemName"] == "LEGAL WRAP")
    & (result["StoreId"] == 963280)
    & (result["OrderType"].isin(["UEats-Delivery"]))
].sort_values(by=["OrderType", "WeekStartDate"])


result[52:100]


def show_graph(filtered_df, weeks, months):
    fig = px.line(
        filtered_df,
        x="WeekStartDate",
        y="item_weekly_volumn",
        color="MenuItemName",  # Highlights decline segment vs normal
        facet_col="OrderType",
        facet_row="StoreId",
        markers=True,
        title=f"Consective {weeks}-Week Declining Volume Over Last {months}-Months",
    )

    # Clean subplot labels
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        annotations=[
            a for a in fig.layout.annotations if a.text not in ["StoreId", "OrderType"]
        ],
        yaxis_title=None,
    )
    fig.update_yaxes(matches=None)
    fig.write_image(
        f"./output/decline_{weeks}_weeks_{months}_months.png", width=1200, height=800
    )
    fig.show()


def x_weeks_consecutive_decline_y_month(df, weeks=4, months=3, show_graphs=True):
    """
    Flags each week's data for menu items that had `weeks` consecutive declines
    in the last `months` months.

    Returns original DataFrame with an added column 'DeclineFlag' = 'decline' or 'normal'
    """

    df_recent = df.sort_values(
        ["StoreId", "OrderType", "MenuItemName", "WeekStartDate"]
    )

    # Initialize flag column
    df_recent["DeclineFlag"] = "normal"

    # Define logic to find and flag X consecutive declining weeks
    def flag_consecutive_declines(group):
        vols = group["item_weekly_volumn"].values
        flags = ["normal"] * len(vols)
        count = 0

        for i in range(1, len(vols)):
            if vols[i] < vols[i - 1]:
                count += 1
                if count >= weeks:
                    # Mark the last `weeks` rows as 'decline'
                    for j in range(i - weeks + 1, i + 1):
                        flags[j] = "decline"
            else:
                count = 0

        group["DeclineFlag"] = flags
        return group

    # Apply per item
    flagged_df = df_recent.groupby(
        ["StoreId", "OrderType", "MenuItemName"], group_keys=False
    ).apply(flag_consecutive_declines)
    # Filter for rows where is_decline_segment == 'decline'
    decline_rows = flagged_df[flagged_df["DeclineFlag"] == "decline"]

    # Get unique combinations that have any decline
    decline_combinations = decline_rows[
        ["StoreId", "OrderType", "MenuItemName"]
    ].drop_duplicates()

    # Now filter the original DataFrame to only include those combinations
    filtered_df = flagged_df.merge(
        decline_combinations, on=["StoreId", "OrderType", "MenuItemName"], how="inner"
    )

    if show_graphs:
        show_graph(filtered_df, weeks, months)

    return filtered_df


weeks = 3
months = 3
show_graphs = True
week_4_consective_decline = x_weeks_consecutive_decline_y_month(
    high_volume_items, weeks, months, show_graphs
)


high_volume_items.groupby(["StoreId", "OrderType", "MenuItemName"])[
    "item_weekly_volumn"
].sum().reset_index().sort_values(by="item_weekly_volumn", ascending=False)
week_4_consective_decline["MenuItemName"].unique()
week_4_consective_decline[
    (week_4_consective_decline["MenuItemName"] == "PARADISE ACAI BOWL")
    & (week_4_consective_decline["StoreId"] == 963290)
    & (week_4_consective_decline["OrderType"].isin(["Take Out", "Online-Dispatch"]))
].sort_values(by=["OrderType", "WeekStartDate"])
#


# PARADISE ACAI BOWL has declined for consective 3 weeks in 963290 store for Take Out and Online-Dispatch orders
# dropping from 15 to 2 on Take Out and  67 to 18 on Online-Dispatch orders
week_4_consective_decline[
    (week_4_consective_decline["MenuItemName"] == "POKE BOWL")
    & (week_4_consective_decline["StoreId"] == 963290)
    & (week_4_consective_decline["OrderType"].isin(["Online-Pickup"]))
].sort_values(by=["OrderType", "WeekStartDate"])

week_4_consective_decline[
    (week_4_consective_decline["MenuItemName"] == "POKE BOWL")
    & (week_4_consective_decline["StoreId"] == 963290)
    & (week_4_consective_decline["OrderType"].isin(["Online-Pickup"]))
].sort_values(by=["OrderType", "WeekStartDate"])["item_weekly_volumn"]


# POKE BOWL sales through Online Pickup at West Kendall decreased from an average of 25 units per week (April 28–May 12, 2025) to 10 units per week (May 19–June 2, 2025) — a 58% decrease.
def percetage_change(old, new):
    if old == 0:
        return float("inf")  # Avoid division by zero

    return ((new - old) / old) * 100


old = (28 + 27 + 21) / 3
new = (12 + 15 + 5) / 3
p_change = percetage_change(old, new)
print(f"Percentage change from {old} to {new} is {p_change:.2f}%")


weeks = 4
months = 3
show_graphs = True
week_4_consective_decline = x_weeks_consecutive_decline_y_month(
    high_volume_items, weeks, months, show_graphs
)
week_4_consective_decline.MenuItemName.unique()
week_4_consective_decline[
    (week_4_consective_decline["MenuItemName"] == "ALMOND BUTTER ACAI BOWL")
    & (week_4_consective_decline["StoreId"] == 963290)
    & (week_4_consective_decline["OrderType"].isin(["Take Out"]))
].sort_values(by=["OrderType", "WeekStartDate"])


def icreasing_trend():
    df = high_volume_items.copy()

    # Prepare a results list
    results = []

    # Group by StoreId and OrderType
    for (store_id, order_type), group in df.groupby(["StoreId", "OrderType"]):
        trend_data = []

        # Further group by MenuItemName
        for menu_item, item_group in group.groupby("MenuItemName"):
            item_group = item_group.sort_values("WeekStartDate")

            # Encode time as numeric (e.g., week number)
            item_group["week_num"] = (
                item_group["WeekStartDate"] - item_group["WeekStartDate"].min()
            ).dt.days

            # Need at least 4 data points to compute a meaningful trend
            if len(item_group) >= 3:
                X = item_group["week_num"].values.reshape(-1, 1)
                y = item_group["item_weekly_volumn"].values
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]

                # Append slope
                trend_data.append((menu_item, slope))

        # Sort by most positive slope (strongest upward trend)
        trend_data.sort(key=lambda x: x[1], reverse=True)

        # Take top 5 increasing items
        top_5_positive = trend_data[:5]

        for menu_item, slope in top_5_positive:
            results.append(
                {
                    "StoreId": store_id,
                    "OrderType": order_type,
                    "MenuItemName": menu_item,
                    "TrendSlope": slope,
                }
            )

    # Convert results to DataFrame
    top_increasing_items = pd.DataFrame(results)
    top_increasing_items
    df_tren_pos = df.merge(
        top_increasing_items, on=["StoreId", "OrderType", "MenuItemName"]
    )
    df_tren_pos
    # Step 1: Filter only items with po TrendSlope
    pos_trend = df_tren_pos[df_tren_pos["TrendSlope"] > 0]
    print("number of item in increasing trens", pos_trend.MenuItemName.unique())

    viz(pos_trend)
    return pos_trend


pos_trend = icreasing_trend()
pos_trend.OrderType.unique()
# report back on last weeks declined items
# : The weekly order volume of the LIV Wrap declined by 86% on Uber Eats at the West Kendall (London Square) location.

pos_trend[
    (pos_trend["MenuItemName"] == "LIV WRAP")
    & (pos_trend["StoreId"] == 963290)
    & (pos_trend["OrderType"].isin(["UEats-Delivery"]))
].sort_values(by=["OrderType", "WeekStartDate"])

pos_trend[
    (pos_trend["MenuItemName"] == "LIV WRAP")
    & (pos_trend["StoreId"] == 963280)
    & (pos_trend["OrderType"].isin(["UEats-Delivery"]))
].sort_values(by=["OrderType", "WeekStartDate"])


def insight(result):
    # Filter and sort data
    df = result[
        (result["MenuItemName"] == "LEGAL WRAP")
        & (result["StoreId"] == 963260)
        & (result["OrderType"] == "UEats-Delivery")
    ].sort_values("WeekStartDate")

    # Group data
    groups = df.groupby(["StoreId", "OrderType"])
    n = len(groups)

    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        specs=[[{"secondary_y": True}] for _ in range(n)],
        # subplot_titles=[f"Store {store} - {order}" for (store, order) in groups.groups.keys()],
        vertical_spacing=0.12,
    )

    for i, ((store, order), group) in enumerate(groups, start=1):
        group = group.sort_values("WeekStartDate")

        # Prepare trendline
        x_ord = group["WeekStartDate"].map(pd.Timestamp.toordinal)
        slope = group["TrendSlope"].iloc[0]
        intercept = group["item_weekly_volumn"].iloc[0] - slope * x_ord.iloc[0]
        trend_y = slope * x_ord + intercept

        # Trendline (blue dashed)
        fig.add_trace(
            go.Scatter(
                x=group["WeekStartDate"],
                y=trend_y,
                mode="lines",
                name="Trendline",
                line=dict(color="royalblue", dash="dash", width=2),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=False,
        )

        # Volume line with labels slightly above
        fig.add_trace(
            go.Scatter(
                x=group["WeekStartDate"],
                y=group["item_weekly_volumn"],
                mode="lines+markers+text",
                name="Volume",
                line=dict(color="royalblue", width=2),
                marker=dict(size=6),
                # text=[str(v) for v in group['item_weekly_volumn']],
                textposition="top center",
                textfont=dict(size=10),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=False,
        )

        # Revenue line with labels further below
        fig.add_trace(
            go.Scatter(
                x=group["WeekStartDate"],
                y=group["item_weekly_revenue"],
                mode="lines+markers+text",
                name="Revenue",
                line=dict(color="forestgreen", width=2),
                marker=dict(size=6),
                # text=[f"${v:,.0f}\n" for v in group['item_weekly_revenue']],  # add newline for spacing hack
                # textposition='bottom center',
                textfont=dict(size=10),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=True,
        )
        fig.add_annotation(
            text=f"Slope: {slope:.2f}",
            xref="paper",
            yref="paper",
            x=0.95,
            y=1 - (i - 1) / n + 0.05,
            showarrow=False,
            font=dict(color="gray", size=11),
            align="right",
        )

    # Layout settings
    fig.update_layout(
        height=450 * n,
        width=750,
        title_text="Weekly Volume & Revenue with Trendline for LEGAL WRAP (Store 963260 - UEats Delivery)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=60),
    )
    fig.update_layout(
        title_font=dict(size=16),  # smaller main title font
        legend_font=dict(size=10),  # smaller legend font
        font=dict(size=10),  # default font size for tick labels, axis titles, etc.
    )

    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))

    # Axis titles and formatting
    fig.update_yaxes(
        title_text="Volume", secondary_y=False, zeroline=True, zerolinecolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Revenue ($)",
        secondary_y=True,
        tickprefix="$",
        zeroline=True,
        zerolinecolor="lightgray",
    )

    fig.update_xaxes(title_text="Week Start Date", tickformat="%Y-%m-%d")
    fig.update_layout(
        title={
            "text": "Weekly Volume & Revenue with Trendline for LEGAL WRAP (Store 963260 - UEats Delivery)",
            "font": {"size": 14},  # smaller size here (default is bigger)
            "x": 0.5,  # center title horizontally
            "xanchor": "center",
        }
    )

    fig.show()



'''Time period covered: 2025-04-28 to 2025-06-02
1. The average weekly volume of LEGAL WRAP at the Weston store on UEats-Delivery has decreased by 47%, dropping from an average of 106 units to 56 units over the period from 2025-04-28 to 2025-06-02.
2. PARADISE ACAI BOWL has seen a significant decline over 3 consecutive weeks at Store West Kendall (London Square) with Take Out orders dropping from 15 to 2 and Online-Dispatch orders falling from 67 to 18.
3. POKE BOWL sales through Online Pickup at West Kendall decreased from an average of 25 units per week (April 28–May 12) to 10 units per week (May 19–June 2) — a 58% decrease.


4. Legal Wrap, Chicken Caesar Wrap, and Poke Bowl consistently rank as the top 3 items by revenue share across all three stores.  LIV Wrap holds the 4th position at Pinecrest and Weston, but ranks 5th in West Kendall.
5. LIV WRAP sales via UEats-Delivery at Pinecrest grew from 16 units ($349.16) during the week starting April 28  to 75 units ($1,754.25) in the week starting June 2.'''