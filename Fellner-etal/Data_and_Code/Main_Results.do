capture clear all
set more off
set mem 600m
set matsize 800

/*     Put the data "data_final.dta" into a folder and     */
/* !!! ADJUST THE FOLDER DIRECTORY IN THE NEXT COMMAND !!! */

cd "C:\DATA\GIS_feldexp\"


use "data_final.dta",clear

******************************************************************************

global Z  schober pop_density2005 pop2005  nat_EU nat_nonEU  fam_marri fam_divor_widow    rel_evan rel_isla rel_orth_other rel_obk    pers2-pers4 pers5more  vo_r vo_cl vo_l  j_unempl j_retire j_house j_studen     inc_aver edu_aver age_aver   bgld kaern noe ooe salzbg steierm tirol vlbg
global Z_extended pop_density2005 pop2005  nat_EU nat_nonEU  fam_marri fam_divor_widow    edu_hi edu_lo  rel_evan rel_isla rel_orth_other rel_obk    pers2-pers4 pers5more  vo_r vo_cl vo_l  j_unempl j_retire j_house j_studen     inc_aver age0_30 age30_60  bgld kaern noe ooe steierm tirol vlbg

******************************************************************************

/* Section 3.2 -- Table 1 */

su gender age pop2005 pop_density2005 compliance

bysort treatment: su gender age pop2005 pop_density2005 compliance

******************************************************************************

/* Section 5.1  -- Mailing Effect  */

*registrations during day 0-25, 25-50, 50-75, 75-100 after sending of mailing

gen resp_A_2550=resp_A
replace resp_A_2550=0 if resp_A_25==1

gen resp_A_5075=resp_A_75
replace resp_A_5075=0 if resp_A==1

gen resp_A_75100=resp_A_100
replace resp_A_75100=0 if resp_A_75==1

* Tests

tab mailing resp_A_25,row
prtest resp_A_25, by(mailing)

tab mailing resp_A_2550,row
prtest resp_A_2550, by(mailing)

tab mailing resp_A_5075,row
prtest resp_A_5075, by(mailing)

tab mailing resp_A_75100,row
prtest resp_A_75100, by(mailing)


*cummulated registrations after 50 day
tab mailing resp_A,row
prtest resp_A, by(mailing)


******************************************************************************

/* Section 5.2  -- Figure 1 */

*A) registrations
bysort treatment: tab resp_A if delivered==1
*graph bar resp_A if delivered==1, over(treatment)

*B) update responses
bysort treatment: tab resp_B if delivered==1

*C) overall response rate
bysort treatment: tab resp_all if delivered==1


/* Section 5.2  -- Tests */

* A) registrations (same sequence as in the text)

prtest resp_A if delivered==1 & (treatment==1|treatment==2), by(threat)
prtest resp_A if delivered==1 & (treatment==3|treatment==4), by(threat)
prtest resp_A if delivered==1 & (treatment==5|treatment==6), by(threat)

prtest resp_A if delivered==1 & (treatment==1|treatment==3), by(info)
prtest resp_A if delivered==1 & (treatment==1|treatment==5), by(appeal)

prtest resp_A if delivered==1 & (treatment==2|treatment==4), by(info)
prtest resp_A if delivered==1 & (treatment==2|treatment==6), by(appeal)


* B) update responses  (same sequence as in the text)

prtest resp_B if delivered==1 & (treatment==1|treatment==2), by(threat)
prtest resp_B if delivered==1 & (treatment==3|treatment==4), by(threat)
prtest resp_B if delivered==1 & (treatment==5|treatment==6), by(threat)

prtest resp_B if delivered==1 & (treatment==1|treatment==5), by(appeal)
prtest resp_B if delivered==1 & (treatment==2|treatment==6), by(appeal)

prtest resp_B if delivered==1 & (treatment==1|treatment==3), by(info)
prtest resp_B if delivered==1 & (treatment==2|treatment==4), by(info)


*C) overall response rate (same sequence as in the text)

prtest resp_all if delivered==1 & (treatment==1|treatment==2), by(threat)
prtest resp_all if delivered==1 & (treatment==3|treatment==4), by(threat)
prtest resp_all if delivered==1 & (treatment==5|treatment==6), by(threat)

prtest resp_all if delivered==1 & (treatment==1|treatment==3), by(info)
prtest resp_all if delivered==1 & (treatment==2|treatment==4), by(info)

prtest resp_all if delivered==1 & (treatment==1|treatment==5), by(appeal)
prtest resp_all if delivered==1 & (treatment==2|treatment==6), by(appeal)



/* Section 5.2  -- Figure 2 */

di "see file: Time patterns.xlsx"


******************************************************************************

/* Section 5.3  -- De-registrations */

tab deregistration

graph bar deregistration if mailing==1, over(treatment)

prtest deregistration, by(threat)


******************************************************************************

/* Section 5.4  -- Table 2 */

reg resp_A mailing threat appeal info, robust
reg resp_A mailing threat appeal info i_tinf i_tapp, robust

reg resp_B    threat appeal info if delivered==1, robust
reg resp_B    threat appeal info i_tinf i_tapp if delivered==1, robust
test i_tinf i_tapp
test threat+i_tapp=0

reg resp_all   threat appeal info if delivered==1, robust
reg resp_all   threat appeal info i_tinf i_tapp if delivered==1, robust


******************************************************************************

/* Section 5.5  -- Table 3 */


reg resp_A  threat appeal info threat_evasion_D1 appeal_evasion_D1 info_evasion_D1 evasion_1 gender $Z if delivered==1, robust
*outreg2 threat appeal info threat_evasion_D1 appeal_evasion_D1 info_evasion_D1 evasion_1  using TAB3_fin,  bdec(3) replace
test info +info_evasion_D1=0

reg resp_A   threat appeal info threat_evasion_D2 appeal_evasion_D2 info_evasion_D2 evasion_2 gender $Z if delivered==1, robust
*outreg2 threat appeal info threat_evasion_D2 appeal_evasion_D2 info_evasion_D2 evasion_2  using TAB3_fin,  bdec(3) append
test info+info_evasion_D2=0




******************************************************************************

/* ONLINE APPENDIX  -- Section C */

******************************************************************************

/* Section C  -- Table C1 */

keep if delivered==1
count


su pop2005,d
gen     pop_hi=0
replace pop_hi=1 if pop2005>=r(p50) & pop2005~=.

su pop_density2005,d
gen     popdens_hi=0
replace popdens_hi=1 if pop_density2005>=r(p50) & pop_density2005~=.

su inc_aver,d
gen     inc_hi=0
replace inc_hi=1 if inc_aver> r(p50) & inc_aver~=.

gen vo_right=vo_cr+vo_r
su vo_right

su vo_right,d
gen     right_hi=0
replace right_hi=1 if vo_right>=r(p50) & vo_right~=.


reg resp_A threat appeal info   gender compliance_t  $Z_extended  if pop_hi==1, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex replace
reg resp_A threat appeal info   gender compliance_t  $Z_extended  if pop_hi==0, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append

reg resp_A threat appeal info   gender compliance_t  $Z_extended  if popdens_hi==1, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append
reg resp_A threat appeal info   gender compliance_t  $Z_extended  if popdens_hi==0, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append

reg resp_A threat appeal info   gender compliance_t  $Z_extended  if inc_hi==1, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append
reg resp_A threat appeal info   gender compliance_t  $Z_extended  if inc_hi==0, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append

reg resp_A threat appeal info   gender compliance_t  $Z_extended  if right_hi==1, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append
reg resp_A threat appeal info   gender compliance_t  $Z_extended  if right_hi==0, robust
* outreg2 threat appeal info     using Table_C1_Online_Appendix, bdec(3) tex append
