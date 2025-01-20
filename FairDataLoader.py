import pandas as pd
from scipy.io import arff

def _str_replace(string):
    string = string.replace('<', '_lt_')
    string = string.replace('[', '_')
    string = string.replace(']', '_')
    string = string.replace(' ', '_')
    return string
    
def get_dutch_census_data():
    '''
    ref:
    Van der Laan, P. (2000). The 2001 census in the netherlands. In Conference the Census of Population

    Le Quy, T., Roy, A., Friege, G., & Ntoutsi, E. (2021). Fair-capacitated clustering. 
    In Proceedings of the 14th International Conference on Educational Data Mining (EDM21). (pp. 407-414).
    '''
    data = arff.loadarff('data/dutch_census/dutch_census_2001.arff')
    df = pd.DataFrame(data[0])
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x))
        df = pd.get_dummies(df, columns=[col], prefix = [col], drop_first=True)
    X = df.iloc[:, :50]
    X.columns = [_str_replace(col) for col in X.columns]
    y = df['occupation_549']
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_adult_data():
    '''
    ref:
    Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
    '''
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
            'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 
            'capital_loss', 'hours_per_week', 'native_country', 'y']
    data = pd.read_csv('data/adult/adult.data', na_values=[' ?'], names=cols)
    data2 = pd.read_csv('data/adult/adult.test', na_values=[' ?'], names=cols)
    data = pd.concat((data, data2), axis=0)
    data = data.dropna()
    data['y'] = data['y'] == ' >50K'
    data['y'] = data['y'].apply(lambda x: int(x))
    data = data.reset_index(drop=True)
    data = data.drop('fnlwgt', axis=1)
    data = data.drop('education_num', axis=1) # This step was not advised in the survey paper but this column appears to be repetitve of education so not sure why I'd keep it?
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_german_credit_data():
    '''
    ref:
    Hofmann,Hans. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
    '''
    cols = ['status_of_existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 
    'installment_rate_in_percentage_of_disposable_income', 'personal_status_and_sex',
    'other_debtors_guarantors', 'present_residence_since', 'property', 'age_in_years',
    'other_installment_plans', 'housing', 'number_of_existing_credits_at_this_bank',
    'job', 'number_of_people_being_liable_to_provide_maintenance_for', 'telephone', 'foreign_worker', 'y']
    data = pd.read_table('data/german_credit_data/german.data', sep=' ',  names=cols)
    data['y'] = data['y'] == 1
    data['y'] = data['y'].apply(lambda x: int(x))
    data['marital_status'] = data['personal_status_and_sex'].apply(lambda x: int(x in ['A91', 'A92']))
    data['sex'] = data['personal_status_and_sex'].apply(lambda x: int(x in ['A91', 'A93', 'A94']))
    data = data.drop('personal_status_and_sex', axis=1)
    cat_cols = ['status_of_existing_checking_account', 'credit_history', 'purpose',
    'savings_account_bonds', 'present_employment_since', 'other_debtors_guarantors', 'property',
    'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_bank_marketing_data(): # This does not touch bank-additional since that is what the survey paper appears to do 
    '''
    ref:
    Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
    '''
    data = pd.read_table('data/bank_marketing/bank-full.csv', sep=';')
    data['y'] = data['y'] == 'yes'
    data['y'] = data['y'].apply(lambda x: int(x))
    
    data['default'] = data['default'] == 'yes'
    data['default'] = data['default'].apply(lambda x: int(x))

    data['housing'] = data['housing'] == 'yes'
    data['housing'] = data['housing'].apply(lambda x: int(x))

    data['loan'] = data['loan'] == 'yes'
    data['loan'] = data['loan'].apply(lambda x: int(x))
    
    cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_credit_card_data():
    '''
    ref:
    Yeh,I-Cheng. (2016). default of credit card clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
    '''
    data = pd.read_csv('data/credit_card_clients/default of credit card clients.csv')
    data['SEX'] = data['SEX'] == 1
    data['SEX'] = data['SEX'].apply(lambda x: int(x))
    cat_cols = ['EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_oulad_data():
    '''
    ref:
    Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. 
    Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).
    '''
    data = pd.read_csv('data/oulad/studentInfo.csv')
    data['gender'] = (data['gender'] == 'M').astype(int)
    data['disability'] = (data['disability'] == 'Y').astype(int)
    data = data.drop('id_student', axis=1)
    data = data.dropna()

    cat_cols = ['code_module', 'code_presentation', 'region', 'highest_education', 'imd_band', 'age_band']
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, drop_first=True)

    data = data[(data['final_result'] == 'Pass') | (data['final_result'] == 'Fail') | (data['final_result'] == 'Distinction')]
    data['final_result'] = ((data['final_result'] == 'Pass') | (data['final_result'] == 'Distinction')).astype(int)

    y = data['final_result']
    X = data.drop('final_result', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_lawschool_data():
    '''
    ref:
    Wightman, L. F. (1998). LSAC national longitudinal bar passage study. LSAC research report series.
    
    Le Quy, T., Roy, A., Friege, G., & Ntoutsi, E. (2021). Fair-capacitated clustering. In Proceedings 
    of the 14th International Conference on Educational Data Mining (EDM21). (pp. 407-414).
    '''
    data = pd.read_csv('data/lawschool/law_dataset.csv')
    data['fulltime'] = (data['fulltime'] == 1).astype(int)
    data['race'] = (data['race'] == 'White').astype(int)
    data['male'] = (data['male'] == 1).astype(int)
    data['pass_bar'] = (data['pass_bar'] == 1).astype(int)

    cat_cols = ['fam_inc', 'tier']
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, drop_first=True)

    y = data['pass_bar']
    X = data.drop('pass_bar', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_kdd_census_data():
    cols = ['age', 'class_of_worker', 'detailed_industry_recode', 'detailed occupation recode',
    'education', 'wage_per_hour', 'enroll_in_edu_inst_last_wk', 'marital_stat',
    'major_industry_code', 'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_a_labor_union', 'reason_for_unemployment', 'full_or_part_time_employment_stat',
    'capital_gains', 'capital_losses', 'dividends_from_stocks', 'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat', 'detailed_household_summary_in_household',
    'instance_weight', 'migration_code_change_in_msa', 'migration_code_change_in_reg', 'migration_code_move_within_reg',
    'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
    'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother',
    'country of birth self', 'citizenship', 'own_business_or_self_employed', 'fill_inc_questionnaire_for_veterans_admin',
    'veterans_benefits', 'weeks_worked_in_year', 'year', 'y']
    data = pd.read_csv('data/kdd_census_income/census-income.data.gz', na_values=[' ?', ' NA'],  names=cols) # The survey treats ' NA' as missing values but it's unclear to me if this is accurate
    data2 = pd.read_csv('data/kdd_census_income/census-income.test.gz', na_values=[' ?', ' NA'],  names=cols)
    # data = data.append(data2)
    data = pd.concat((data, data2), axis=0)
    data = data.drop(['instance_weight', 'migration_code_change_in_msa', 'migration_code_change_in_reg', 'migration_code_move_within_reg','migration_prev_res_in_sunbelt'],axis=1)
    data = data.dropna()
    data['y'] = data['y'] == ' 50000+.'
    data['y'] = data['y'].apply(lambda x: int(x))
    data = data.reset_index(drop=True)
    cat_cols = ['class_of_worker', 'detailed_industry_recode', 'detailed occupation recode',
    'education', 'enroll_in_edu_inst_last_wk', 'marital_stat',
    'major_industry_code', 'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_a_labor_union', 'reason_for_unemployment', 'full_or_part_time_employment_stat',
    'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat', 'detailed_household_summary_in_household',
    'live_in_this_house_1_year_ago', 'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother',
    'country of birth self', 'citizenship', 'own_business_or_self_employed', 'fill_inc_questionnaire_for_veterans_admin',
    'veterans_benefits', 'year']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_diabetes_data():
    data = pd.read_csv('data/diabetes/diabetic_data.csv', na_values=['?'])
    data = data.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1)
    data = data[data['readmitted']!='NO']
    data = data.dropna()

    data['change'] = data['change'] == 'Ch'
    data['change'] = data['change'].apply(lambda x: int(x))

    data.loc[data['gender']=='Male', 'gender'] = 1
    data.loc[data['gender']=='Female', 'gender'] = 0
    data['gender'] = data['gender'].astype(int)


    data.loc[data['diabetesMed']=='Yes', 'diabetesMed'] = 1
    data.loc[data['diabetesMed']=='No', 'diabetesMed'] = 0
    data['diabetesMed'] = data['diabetesMed'].astype(int)

    cat_cols = ['race', 'age', 'A1Cresult', 'metformin', 'chlorpropamide', 'glipizide', 'rosiglitazone', 'acarbose', 
    'miglitol', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3',
    'max_glu_serum', 'repaglinide', 'nateglinide', 'glimepiride', 'acetohexamide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
    'metformin-rosiglitazone', 'metformin-pioglitazone']
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, drop_first=True)
    data['readmitted'] = (data['readmitted'] == '<30').astype(int)
    y = data['readmitted']
    X = data.drop('readmitted', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_compas_data():
    main = pd.read_csv('data/COMPAS/compas-scores-two-years.csv')
    main = main[(abs(main['days_b_screening_arrest']) <= 30)]
    cols = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'score_text', 'v_score_text', 'is_recid']
    
    df = main[cols]
    df['y'] = df['is_recid']
    df = df.drop('is_recid', axis=1)
    df.drop(df.index[df['y'] == -1], axis=0).dropna().reset_index(drop=True)
    cat_cols = ['sex', 'age_cat', 'race', 'c_charge_degree', 'score_text', 'v_score_text']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    y = df.y
    X = df.drop('y', axis=1)
    return X.reset_index(drop=True), y.reset_index(drop=True)