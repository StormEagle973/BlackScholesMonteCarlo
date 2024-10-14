# Black-Scholes and Monte Carlo Option Pricer
This Flask application allows users to explore European cash-settled option pricing using the **Black-Scholes formula** and **Monte Carlo simulations**.
The app provides interactive tools to calculate option prices and visualize the data through heatmaps and distribution plots, and is a useful tool for 
insight on the mechanics of option pricing and vizualizing outcomes for option prices with varying parameters. 

https://tthunga24.pythonanywhere.com/

### Features
- **Black-Scholes Option Pricing:** Explore how European cash-settled options are priced by customizing the parameters of the Black-Scholes equation. View call and put prices for each desired set of inputs.

- **Customizable Heatmaps:** Vizualize how prices of calls and puts change based on varying spot prices and a varying paramaeter within a range of the user's choice, with other parameters being held constant to the values specified by the user. (Spot min and Spot max should be entered as multipliers for the user specified Asset Price, with Spot min <= 1 and Spot max > 1)

- **Monte Carlo Simulations:** Estimate the price of call and put options through Monte Carlo simulations. Users can customize the number of samples (n-value) and view:
  - Estimated call and put price.
  - A distribution of the asset price at maturity using Brownian motion.
  - A distribution of the expected call and put payouts at expiration.
 
 ### Technologies/Dependencies
  - `flask`: Web framework for building app
  - `numpy, scipy`: For numerical computations and probability distributions.
  - `matplotlib, seaborne`: For generating dynamic vizualizations such as heatmaps and histograms.
