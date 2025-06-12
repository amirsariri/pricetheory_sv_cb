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

local csvfiles: dir "$DataPath/250612_cb_data" files "*.csv"

foreach filename of local csvfiles {
	import delimit "$DataPath/1raw/`filename'", varn(1) clear
	local filename: subinstr local filename ".csv" ""
	sa "`filename'.dta", replace
}
