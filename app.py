# Imports
import pandas as pd

import streamlit as st
from defectDetect.predict import prediction

st.set_page_config(
    layout="wide",  # Choose your desired layout
)

# Background Style
page_bg_img = """
<style>
[data-testid='stAppViewContainer'] {
background-image: url("https://static.vecteezy.com/system/resources/previews/014/302/734/non_2x/luxury-abstract-banner-template-with-gold-blended-waves-on-black-background-with-copy-space-for-text-shiny-dynamic-light-stripes-futuristic-twisted-lines-design-in-dark-background-wallpaper-website-vector.jpg");
background-size: cover;
}
[data-testid='stSidebar'] {
background-color: rgba(100, 150, 50, 0.5);
}
[data-testid='block-container'] {
    top-margin: 10px;
    padding-top: 40px;
}
[data-testid='column'] {
background-color: rgba(76, 58, 76, 0.95);
padding-left: 30px;
padding-right: 30px;
padding-top: 20px;
padding-bottom: 20px;
}
[data-testid='baseButton-secondary'] {
height: 50px;
width: 100px;
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Heading
st.write(
    """
# DeFectDeTect
"""
)
st.text("")
st.text("")
st.text("")


# Sidebar User Inputs
def user_input_features():
    st.sidebar.write("## Program Features")
    loc = st.sidebar.number_input("loc")
    vg = st.sidebar.number_input("v(g)")
    evg = st.sidebar.number_input("ev(g)")
    ivg = st.sidebar.number_input("iv(g)")
    n = st.sidebar.number_input("n")
    v = st.sidebar.number_input("v")
    l = st.sidebar.number_input("l")
    d = st.sidebar.number_input("d")
    i = st.sidebar.number_input("i")
    e = st.sidebar.number_input("e")
    b = st.sidebar.number_input("b")
    t = st.sidebar.number_input("t")
    lOCode = st.sidebar.number_input("lOCode")
    lOComment = st.sidebar.number_input("lOComment")
    lOBlank = st.sidebar.number_input("lOBlank")
    locCodeAndComment = st.sidebar.number_input("locCodeAndComment")
    uniq_Op = st.sidebar.number_input("uniq_Op")
    uniq_Opnd = st.sidebar.number_input("uniq_Opnd")
    total_Op = st.sidebar.number_input("total_Op")
    total_Opnd = st.sidebar.number_input("total_Opnd")
    branchCount = st.sidebar.number_input("branchCount")

    data = {
        "loc": loc,
        "v(g)": vg,
        "ev(g)": evg,
        "iv(g)": ivg,
        "n": n,
        "v": v,
        "l": l,
        "d": d,
        "i": i,
        "e": e,
        "b": b,
        "t": t,
        "lOCode": lOCode,
        "lOComment": lOComment,
        "lOBlank": lOBlank,
        "locCodeAndComment": locCodeAndComment,
        "uniq_Op": uniq_Op,
        "uniq_Opnd": uniq_Opnd,
        "total_Op": total_Op,
        "total_Opnd": total_Opnd,
        "branchCount": branchCount,
    }

    features = pd.DataFrame(data, index=["User Input"])
    return features


input_df = user_input_features()
co1, co2 = st.columns(2, gap="large")
with co1:
    st.write(
        """### Attribute Information:
1. **loc             :** numeric % McCabe's line count of code
2. **v(g)            :** numeric % McCabe "cyclomatic complexity"
3. **ev(g)           :** numeric % McCabe "essential complexity"
4. **iv(g)           :** numeric % McCabe "design complexity"
5. **n               :** numeric % Halstead total operators + operands
6. **v               :** numeric % Halstead "volume"
7. **l               :** numeric % Halstead "program length"
8. **d               :** numeric % Halstead "difficulty"
9. **e               :** numeric % Halstead "effort"
11. **b               :** numeric % Halstead
12. **t               :** numeric % Halstead's time estimator
13. **lOCode          :** numeric % Halstead's line count
14. **lOComment       :** numeric % Halstead's count of lines of comments
15. **lOBlank         :** numeric % Halstead's count of blank lines
16. **lOCodeAndComment :** numeric
17. **uniq_Op         :** numeric % unique operators
18. **uniq_Opnd       :** numeric % unique operands
19. **total_Op        :** numeric % total operators
20. **total_Opnd      :** numeric % total operands
21. **branchCount     :** numeric % of the flow graph"""
    )

with co2:
    st.write("### Detector")
    st.write("**Please click the side arrow on the top left to open the feature input sidebar.**")
    if st.button("Detect"):
        with st.spinner(""):
            print(input_df)
            result = prediction(input_df)
            st.write("")
            st.write("")
            st.write("#### &nbsp;Probability of Defect in the Code")
            st.write(f"## &nbsp;&nbsp;{round(result*100, 4)}%")
            st.write("")
            st.write("")
            st.write("")
            st.write(
                """##### Result Analysis Guide:
**Probability Score Between**
- **``0% - 5%``** : No defect (unless random input values)
- **``5% - 10%``** : Very less possibility of defect (unless random input values)
- **``10% - 15%``** : Less possibility of defect (unless random input values)
- **``15% - 20%``** : Fair possibility of defect (unless random input values)
- **``20% - 25%``** : Possible defect
- **``25% - 100%``** : High possibility of defect"""
            )
