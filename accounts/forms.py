from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class LogForm(forms.Form):
    username=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.75rem; "}))
    password=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.75rem; "}))


class Reg(UserCreationForm):
    class Meta:
        model=User
        fields=['first_name','last_name','email','username','password1','password2']   
    first_name=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Firstname","class":"form-control","style":"border-radius: 0.75rem; "}))
    last_name=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Lastname","class":"form-control","style":"border-radius: 0.75rem; "}))    
    username=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.75rem; "}))
    email=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Email","class":"form-control","style":"border-radius: 0.75rem; "}))
    password1=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.75rem; "}))
    password2=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Confirm Password","class":"form-control","style":"border-radius: 0.75rem; "})) 





class PredictionForm(forms.Form):
    text = forms.CharField(
        label='Enter Text:',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 50}),
        required=True,
        help_text='Enter the text you want to classify.'
    )
