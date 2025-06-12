/*==================================================================
Program: 10globals.do
Authors: John and Amir
Created: 250612
Edited:  
Purpose: Creates global variables
===================================================================*/

/*================================================
Independent variables
================================================*/

/*================================================
Outcome variables
================================================*/


/*================================================
Esttab (LaTeX) Options
================================================*/

global Esttab_opt "se nonumbers nonote noobs nocons nobase nogap compress label replace"

global Esttab_statsS "star(* .1 ** .05 *** .01) b(3) stats(r2 N, label("\(R^{2}\)" "Observations") fmt(2 %9.0fc))"

global Esttab_statsT "star(* .1 ** .05 *** .01) b(3) stats(r2 obs, label("\rule{0ex}{3ex}\(R^{2}\)" "Observations") fmt(2 %9.0fc))"

global Esttab_statsW "star(* .1 ** .05 *** .01) b(3) stats(r2 obs ymean, label("\rule{0ex}{3ex}\(R^{2}\)" "Observations" "Mean of DV") fmt(2 %9.0fc 2))"

global Esttab_statsX "star(* .1 ** .05 *** .01) b(3) stats(r2, label("\rule{0ex}{3ex}\(R^{2}\)") fmt(2 %9.0fc 2))"

global Esttab_statsO "star(* .1 ** .05 *** .01) b(3) stats(F2 F1 N, label("F(Sci + Sci X GoodSci = 0)" "F(Bus + Bus X GoodBus = 0)" "Observations") fmt(3 3 %9.0fc))"

global Esttab_statsM "star(* .1 ** .05 *** .01) b(3) stats(F2 F1 ymean, label("F(Sci + Sci X GoodSci = 0)" "F(Bus + Bus X GoodBus = 0)" "Mean of DV") fmt(3 3 3))"

*** worker heterogeneity table options
global Esttab_statsH0 "star(* .1 ** .05 *** .01) b(3) stats(N, label("Observations") fmt(%9.0fc))"

global Esttab_statsHM "star(* .1 ** .05 *** .01) b(3)"

/*================================================
Coefplot options
================================================*/
* style configurations
global Coefplot_opt "byopts(graphregion(margin(b+27 t+27)) xrescale row(1) legend(row(1) pos(1))) xlabel(#3) legend(size(small)) grid(n) rename(c1 = "Sci Info Shock" c2 = "Biz Info Shock") scheme(s1mono) xline(0, lcolor(black) lwidth(thin) lpattern(dash))"
global Coefplot_opt1 "byopts(graphregion(margin(b+27 t+27)) xrescale row(1)) xlabel(#3) legend(size(small)) grid(n) rename(c1 = "+ve sci info" c2 = "-ve sci info" c3 = "+ve biz info" c4 = "-ve biz info") xline(0, lcolor(black) lwidth(thin) lpattern(dash))"
global Coefplot_opt2 "byopts(graphregion(margin(l-6 r-4 b+18 t+18)) xrescale row(1) legend(row(1) pos(1))) legend(size(small)) grid(n) coeflabels(c1 = "Sci info" c2 = `""Sci info +" "Sci info X good sci""' c3 = "Biz info" c4 = `""Biz info +" "Biz info X Good biz""') scheme(s1mono) xline(0, lcolor(black) lwidth(thin) lpattern(dash))"

/*================================================
Outregging globals.
================================================*/
global OR2 "se dec(2) nocons nonotes noaster label ctitle(" ") addtext("") tex(frag) slow($Slow_T)"
global OR2_fstat "se dec(2) nocons nonotes noaster label  ctitle(" ") addtext("") tex(frag) addstat(F-stat on excl instrument, e(widstat)) slow($Slow_T)"
global OR2_addtext "se dec(2) nocons nonotes noaster label ctitle(" ") tex(frag) slow($Slow_T)"

* Three decimal points.
global OR2_dec3  "se dec(3) nocons nonotes noaster label ctitle(" ") addtext("") tex(frag) slow($Slow_T)"
global OR2_AT_d3 "se dec(3) nocons nonotes noaster label ctitle(" ") tex(frag) slow($Slow_T)"

* Version that has automatic mean dependent variable.
global OR2_fstatmdv "se dec(2) nocons nonotes noaster label ctitle(" ") addtext("") tex(frag) addstat(F-stat on excl instrument,e(widstat),Mean DV,zz) slow($Slow_T)"
