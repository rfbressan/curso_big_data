capture clear all
set more off
set mem 600m
set matsize 800


/*     Put the data "data_survey.dta" into a folder and     */
/* !!! ADJUST THE FOLDER DIRECTORY IN THE NEXT COMMAND !!! */


cd "C:\DATA\GIS_survey\"



********************************************
use "data_survey.dta",replace
********************************************


global X female age education risk_type use_orf value_fair value_harm value_just value_high value_qual value_system nat_DE nat_IT nat_XX


******************************************************************************

/* ONLINE APPENDIX */

/* Section D.2 -- Survey Results */

******************************************************************************

/* Table D.1 */

reg risk_perception mailing mail_T mail_I mail_M evader   , robust
* outreg2 mailing mail_T mail_I mail_M evader using risk, bdec(3) tex replace
reg risk_perception mailing mail_T mail_I mail_M evader $X, robust
* outreg2 mailing mail_T mail_I mail_M evader using risk, bdec(3) tex append

reg response_propensity mailing mail_T mail_I mail_M evader   , robust
* outreg2 mailing mail_T mail_I mail_M evader using risk, bdec(3) tex append
reg response_propensity mailing mail_T mail_I mail_M evader $X, robust
* outreg2 mailing mail_T mail_I mail_M evader using risk, bdec(3) tex append

******************************************************************************


/* non-parametric analysis */

/* expected fines */

ranksum expected_fine if treatment~=0, by(evader)
ranksum expected_fine if treatment~=0, by(threat)
ranksum expected_fine if treatment~=0, by(info)
ranksum expected_fine if treatment~=0, by(appeal)

/* expected social sanctions */

ranksum expected_social_sanction if treatment~=0, by(evader)
ranksum expected_social_sanction if treatment~=0, by(threat)
ranksum expected_social_sanction if treatment~=0, by(info)
ranksum expected_social_sanction if treatment~=0, by(appeal)
