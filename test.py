
from django import forms


def input(gender,status,dependents,education,selfemp,income,cincome,loan_amount,loan_amount_term,credit_history,property_area):
    test_gender=gender
    test_married=status
    test_dependents=dependents
    test_education=education
    test_self_employed=selfemp
    test_ApplicantIncome=float(income)
    test_CoapplicantIncome=float(cincome)
    test_LoanAmount=float(loan_amount)
    test_LoanAmountTerm=float(loan_amount_term)
    test_creditHistory=int(credit_history)
    test_PropertyArea=property_area
    test_LoanID = "LP00888"
    test_Type = "Test"

    return test_gender,test_married,test_dependents,test_education,test_self_employed,test_ApplicantIncome,test_CoapplicantIncome,test_LoanAmount,test_LoanAmountTerm,test_creditHistory,test_PropertyArea



