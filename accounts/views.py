from django.shortcuts import render,redirect
from .models import *
from .forms import *
from django.views.generic import FormView,CreateView,TemplateView
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.urls import reverse_lazy
# Create your views here.
import keras


class LoginView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        log_form=LogForm(data=request.POST)
        if log_form.is_valid():  
            us=log_form.cleaned_data.get('username')
            ps=log_form.cleaned_data.get('password')
            user=authenticate(request,username=us,password=ps)
            if user: 
                login(request,user)
                return redirect('h')
            else:
                return render(request,'login.html',{"form":log_form})
        else:
            return render(request,'login.html',{"form":log_form}) 
        

class RegView(CreateView):
     form_class=Reg
     template_name="reg.html"
     model=User
     success_url=reverse_lazy("log")    



import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def Home(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        # user_input = request.POST['user_input']  # Get user input from the form
    #     if user_input:
    #          result = predict_sentiment(user_input)  # Make predictions using your model
    #     else:
    #         pass    
        nltk.download("stopwords")
        nltk.download("punkt")

        # Sample data (you should replace this with a larger, labeled dataset)
        data = [
            ("I love this product! good beautiful","No Hate speech detected"),      # Not hate speech (label 0)
            ("You are the worst! bastard shit ", "Hate speech detected"),       # Hate speech (label 1)
            ("This is awful! dirty ", "Hate speech detected"),           # Hate speech (label 1)
            ("Great job!", "No Hate speech detected"),               # Not hate speech (label 0)
            ("I hate your attitude.", "Hate speech detected"),    # Hate speech (label 1)
            ("fuck", "Hate speech detected"),    # Hate speech (label 1)
        ]

        # Preprocess the data
        stop_words = set(stopwords.words("english"))
        translator = str.maketrans("", "", string.punctuation)

        def preprocess_text(text):
            if text is None:
                return ""  
            text = text.lower()
            text = text.translate(translator)
            tokens = nltk.word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words]
            return " ".join(filtered_tokens)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

        # Split the data into features (X) and labels (y)
        X = [text for text, label in data]
        y = [label for text, label in data]

        # Vectorize the text data
        X = vectorizer.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear SVM classifier
        classifier = SVC(kernel="linear")
        classifier.fit(X_train, y_train)

        # Predict labels for test data
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Example to predict hate speech
        new_text = text
        new_text = preprocess_text(new_text)
        new_text_vectorized = vectorizer.transform([new_text])
        prediction = classifier.predict(new_text_vectorized)

        if prediction == 1:
            print("Hate speech detected.")
        else:
            print("No hate speech detected.")
        context = {'text': new_text, 'prediction': prediction}
        return render(request, 'home.html', context)
    # return HttpResponse("Invalid request method.")

    # return render(request, 'home.html', {'result': result})




