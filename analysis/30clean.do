/*=====================================
Program: 	20clean.do
Authors: 	John and Amir
Created: 	250612
Edited:  	
Purpose: 	Performs data preparations
======================================*/

clear
set more off

/*=====================================
GUIDE TO THE CODE:
1.	
======================================*/

/*=====================================
1. convert csv to dta
======================================*/


clear
set more off

cd "$DataPath"

local csvfiles: dir "$DataPath/processed" files "orgs_2012_2019_survived.csv"

foreach filename of local csvfiles {
	import delimit "$DataPath/processed/`filename'", varn(1) clear
	local filename: subinstr local filename ".csv" ""
	sa "$DataPath/processed/dta/`filename'.dta", replace
}
