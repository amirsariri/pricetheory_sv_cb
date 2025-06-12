/*==================================================================
Program: 	00master.do
Authors: 	Amir and John
Created: 	250612
Edited:  	
Purpose: 	Runs all analyses
===================================================================*/

/*====================================================================
GUIDE TO THE CODE
1.	
====================================================================*/

clear all

/*====================================================================
1. Set Directory Paths
====================================================================*/
	
/*======================================================
1.1. Set Author's Local Directory
======================================================*/

* Amir Sariri
global Main = "/Users/asariri/Library/CloudStorage/Dropbox/Fun/Research/Amir_John"
// global Main = "/Users/asariri/Dropbox/Fun/Research/Amir_John"

/*======================================================
1.2. Set Path for data, analysis and log files
======================================================*/
	
* .do files for cleaning and analysis
global AnlzPath = "$Main/analysis"

* data folder
global DataPath = "$Main/data"
	
* .log file path
global LogPath = "$Main/analysis/logs"
	
global Ob = "$Main/drafts/ob"


/*======================================================
2. Start a dated log file
======================================================*/
capture log close

* create date and time macros
local date: display %td_YY_NN_DD date(c(current_date), "DMY")
local date_string = subinstr(trim("`date'"), " " , "_", .)
log using "$LogPath/`date_string'_${S_TIME}_svTheory.log", append


/*======================================================
3. Set global macros
======================================================*/
qui do "$AnlzPath/10globals.do"

/*====================================================================
4. Performs data preparations
====================================================================*/
qui do "$AnlzPath/20clean.do"

/*====================================================================
5. Perform main analyses
	1.	
====================================================================*/
qui do "$AnlzPath/30anlysis.do"

log close



