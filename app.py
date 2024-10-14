from flask import Flask, render_template, request, session
from numpy import sqrt, log, exp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sn
import io
import base64
import secrets


app = Flask(__name__)
app.config['SESSION_PERMANENT'] = False
app.secret_key = secrets.token_hex(32)
class test:
    def __init__(self, asset_price = 100.00, strike_price= 100.00, time_to_mature = 1.00, volatility = 0.2, risk_free = 0.05, div_yield = 0.00, nv = 10000,
                 spmin = 0.8, spmax = 1.2, hmp = "time_to_mature", pmin = 0, pmax = 1
                  ):
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.time_to_mature = time_to_mature
        self.volatility = volatility
        self.risk_free = risk_free
        self.div_yield = div_yield
        self.nv = nv
        self.spmin = spmin
        self.spmax = spmax
        self.hmp = hmp
        self.pmin = pmin
        self.pmax = pmax


    def calcBS(self, asset_price=None, strike_price=None, time_to_mature=None, volatility=None, risk_free=None, div_yield=None):

        asset_price = asset_price if asset_price is not None else self.asset_price
        strike_price = strike_price if strike_price is not None else self.strike_price
        time_to_mature = time_to_mature if time_to_mature is not None else self.time_to_mature
        volatility = volatility if volatility is not None else self.volatility
        risk_free = risk_free if risk_free is not None else self.risk_free
        div_yield = div_yield if div_yield is not None else self.div_yield

        d1 = (
              (
            log(asset_price/strike_price) +
            (risk_free - div_yield + (volatility ** 2)/2)*time_to_mature
              ) /
              (volatility * sqrt(time_to_mature))
        )

        d2 = d1 - (volatility * sqrt(time_to_mature))

        CallPrice = (
            (
                asset_price * exp(-1 * div_yield * time_to_mature) * norm.cdf(d1)
            ) -
            (
                strike_price * exp(-1*risk_free*time_to_mature) * norm.cdf(d2)
            )
        )
        PutPrice = (
             (
                strike_price * exp(-1*risk_free*time_to_mature) * norm.cdf(-1 * d2)
            ) -
            (
                asset_price * exp(-1 * div_yield * time_to_mature) * norm.cdf(-1 * d1)
            )

        )




        return [round(CallPrice, 2), round(PutPrice, 2)]



    def calcMC(self):
        asset_price = self.asset_price
        strike_price = self.strike_price
        time_to_mature = self.time_to_mature
        volatility = self.volatility
        risk_free = self.risk_free
        div_yield = self.div_yield
        n = self.nv

        Z = np.random.normal(0, 1, n)
        S_T = asset_price * np.exp((risk_free - div_yield - 0.5 * volatility**2) * time_to_mature + volatility * np.sqrt(time_to_mature) * Z)
        payoffsc = np.maximum(S_T - strike_price, 0)
        payoffsp = np.maximum(strike_price - S_T, 0)
        plt.figure(figsize=(8, 6))
        sn.histplot(S_T)
        plt.ylabel("Count")
        plt.xlabel("Asset Price at Time Tm")
        plt.title("Distribution of Asset Price at Time of Maturity using Brownian Motion")

        img_call = io.BytesIO()
        plt.savefig(img_call, format='png')
        plt.clf()
        img_call.seek(0)
        img_bm_base64 = base64.b64encode(img_call.getvalue()).decode('utf-8')

        plt.figure(figsize=(8, 6))
        sn.histplot(payoffsp)
        sn.histplot(payoffsc)
        plt.ylabel("Count")
        plt.xlabel("Contract Value at Expiration")
        plt.title("Distribution of Contract Values at Expiration")
        plt.legend(["Put","Call"])
        img_call = io.BytesIO()
        plt.savefig(img_call, format='png')
        plt.clf()
        img_call.seek(0)
        img_pc_base64 = base64.b64encode(img_call.getvalue()).decode('utf-8')



        return[np.mean(payoffsc) * exp(-risk_free * time_to_mature), np.mean(payoffsp) * exp(-risk_free * time_to_mature), img_bm_base64, img_pc_base64]




    def heatmap(self):
        param = self.hmp
        smin = self.spmin
        smax = self.spmax
        pmin = self.pmin
        pmax = self.pmax
        hmarrc = np.zeros((8,8))
        hmarrp = np.zeros((8,8))
        sps = np.linspace(self.asset_price*smin,self.asset_price*smax,8)
        ps = np.linspace(pmin,pmax,8)
        for x, spot_price in enumerate(sps):
            for y, pval in enumerate(ps):
                bsparam = {
                    'asset_price': spot_price,
                    # Set the parameter to the current value
                   param: pval
                }
                ans = self.calcBS(**bsparam)
                hmarrc[x][y], hmarrp[x][y] = ans[0],ans[1]

        plt.figure(figsize=(8, 6))
        sn.heatmap(hmarrc.T, annot=True, fmt=".2f", cmap="RdYlGn")
        plt.ylabel(f"{param}")
        plt.xlabel("Spot Price")
        plt.title("Call Price Heatmap")
        plt.yticks(ticks=np.arange(8), labels=np.round(ps, 2))
        plt.xticks(ticks=np.arange(8), labels=np.rint(sps ))
        img_call = io.BytesIO()
        plt.savefig(img_call, format='png')
        plt.clf()
        img_call.seek(0)
        img_call_base64 = base64.b64encode(img_call.getvalue()).decode('utf-8')


        plt.figure(figsize=(8, 6))
        sn.heatmap(hmarrp.T, annot=True, fmt=".2f",cmap="RdYlGn")
        plt.ylabel(f"{param}")
        plt.xlabel("Spot Price")
        plt.title("Put Price Heatmap")
        plt.yticks(ticks=np.arange(8), labels=np.round(ps, 2))
        plt.xticks(ticks=np.arange(8), labels=np.rint(sps ))
        img_put = io.BytesIO()
        plt.savefig(img_put, format='png')
        plt.clf()
        img_put.seek(0)
        img_put_base64 = base64.b64encode(img_put.getvalue()).decode('utf-8')


        return img_call_base64, img_put_base64



@app.before_request
def clear_session():

    session.clear()

@app.route("/",methods=['POST','GET','POSTH'])
def hello_world():
    a=test()
    heatmap_call, heatmap_put, monte_hist, monte_dist, mc,mp = None, None, None, None, None, None
    mc,mp, monte_hist,monte_dist = a.calcMC()
    heatmap_call, heatmap_put = a.heatmap()

    if request.method == 'POST':

        a.asset_price = round(float(request.form["asset_price"]),2)
        a.strike_price = round(float(request.form["strike_price"]),2)
        a.time_to_mature =round(float(request.form["time_to_mature"]),2)
        a.volatility = round(float(request.form["volatility"]),2)
        a.risk_free = round(float(request.form["risk_free"]),2)
        a.div_yield = round(float(request.form["div_yield"]),2)
        a.nv = int(float(request.form["nval"]))
        a.spmin = round(float(request.form["spmin"]),2)
        a.spmax = round(float(request.form["spmax"]),2)
        a.hmp = request.form["hmp"]
        a.pmin = round(float(request.form["pmin"]),2)
        a.pmax = round(float(request.form["pmax"]),2)
        mc,mp, monte_hist,monte_dist = a.calcMC()
        heatmap_call, heatmap_put = a.heatmap()


    return render_template('index.html',sss = a, heatmap_call= heatmap_call, heatmap_put = heatmap_put, monte_hist=monte_hist, monte_dist=monte_dist, mc=mc,mp=mp)
