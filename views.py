from django.shortcuts import render
from subprocess import run
from subprocess import PIPE
import sys
import test
import loan
def buton(request):
    return render(request, 'home.html')


def form_view(request):
    return render(request, 'form.html')

def external(request):
    gender=request.POST.get('gender')
    status=request.POST.get('status')
    dependents=request.POST.get('dependents')
    education=request.POST.get('education')
    selfemp=request.POST.get('selfemp')
    income=request.POST.get('income')
    c_income=request.POST.get('c_income')
    loan_amount=request.POST.get('loan_amount')
    loan_amount_term=request.POST.get('loan_amount_term')
    credit_history= request.POST.get('credit_history')
    property_area=request.POST.get('property_area')
   # out=run([sys.executable,'//test.py',gender,status,dependents,education,selfemp,income,c_income,loan_amount,loan_amount_term,
    #         credit_history,property_area], shell=False, stdout=PIPE)
    output=loan.input(gender,status,dependents,education,selfemp,income,c_income,loan_amount,loan_amount_term,credit_history,property_area)
    if(output=="Y"):
        output="Loan Status: Yes"
    else:
        output="Loan Status: No"
    #print(out)
    return render(request, 'try.html',{'data1': output})