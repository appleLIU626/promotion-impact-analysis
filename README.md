# Promotion Impact Analysis (Tableau + Python)

## Project Overview
This project analyzes the impact of low-price promotions on sales quantity and revenue using retail transaction data.  
The objective is to evaluate whether discounting strategies meaningfully increase purchase volume and to identify product categories where promotions are effective versus detrimental.

The analysis combines **Python-based data preparation** with an **interactive Tableau Public dashboard** to deliver business-oriented insights.

---

## Business Questions
- Do low-price promotions drive higher sales quantity?
- What is the revenue trade-off associated with discounting?
- Which product categories benefit most from low-price strategies?

---

## Data
- **Source:** Public retail transaction dataset (Kaggle)
- **Granularity:** Transaction-level sales records
- **Key fields:**
  - Product line
  - Unit price
  - Quantity
  - Total sales amount
  - Promotion proxy (low-price vs regular-price)

---

## Methodology

### 1. Promotion Proxy
Because explicit promotion indicators were not available, a **low-price proxy** was constructed:
- Transactions priced lower than comparable items within the same product category were labeled as **Low Price**
- All other transactions were labeled as **Regular Price**

This approach enables promotion-effect analysis without relying on campaign metadata.

---

### 2. KPI Comparison
Key performance indicators were compared between Low Price and Regular Price transactions:
- Average quantity per transaction
- Average unit price
- Average revenue per transaction

This step evaluates the trade-off between potential quantity uplift and revenue impact.

---

### 3. Category-Level Analysis
Sales quantity differences were analyzed at the product category level to:
- Identify categories with positive quantity uplift under low-price strategies
- Highlight categories where discounting does not translate into higher volume

This supports targeted, category-specific promotion recommendations.

---

## Tools & Technologies
- **Python** (Pandas, NumPy)
  - Data cleaning and feature engineering
- **Tableau Public**
  - Interactive dashboards
  - KPI comparison
  - Category-level visualization

---

## Key Insights
- Overall quantity uplift from low-price transactions is limited.
- Revenue per transaction decreases significantly under low-price pricing.
- Promotion effectiveness varies by category:
  - *Electronic Accessories* shows positive quantity uplift.
  - *Fashion Accessories* underperforms under low-price strategies.

---

## Dashboard
👉 **Interactive Tableau Public Dashboard:**  
[Add your Tableau Public link here]

The dashboard includes:
- KPI overview comparing Low Price vs Regular Price transactions
- Category-level quantity comparison highlighting promotion effectiveness

---

## Business Recommendations
- Avoid blanket discounting across all product categories.
- Apply promotions selectively to categories that demonstrate positive quantity response.
- Evaluate promotion success using both quantity uplift and revenue impact.

---

## Limitations & Next Steps
- Time-based analysis was not performed due to the absence of date information.
- Future work could incorporate margin data to assess promotion ROI.
- Order-level aggregation may provide deeper insights into basket behavior.

---

## Author
**Shihao Liu**  
Aspiring Data Analyst | Python • SQL • Tableau  
www.linkedin.com/in/shihao-liu-2a611114a
