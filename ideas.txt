1) Obrazek generować dla większego obszaru, a porównywać z oryginałem tylko środkową część,
o odpowiednich wymiarach. w ten sposób zwiekszymy prawdopodobieństo generowania się trójkątów równierz na brzegach

2) Mutować osobno kordynaty a osobno kolory (Jeżeli teraz tak nie jest)

3) Nową populacje opierać na krzyżowaniu a nie na mutowaniu.
    a) Krzyżowanie 2 obrazków polega na stworzeniu nowego przez pobranie połowy trójkątó z pierwszego,
    i połowy trójkątów z drugiego.

4) Mutowanie na razie nie ma sensu z funkcją { new = old + (new - 0.5)*diff }, bo new jest losowym ciągiem,