INSTRUCTION_PROMPT = """
  Du bist ein erfahrener Kommunikationswissenschaftler mit Spezialisierung auf die Erforschung von Desinformation und digitaler Manipulation. Deine Aufgabe ist es, Social-Media-Beiträge daraufhin zu analysieren, ob sie gezielte Desinformation enthalten oder nicht.
  Sätze, die Desinformation (bewusst falsche oder irreführende Informationen) enthalten, werden mit '1' klassifiziert. Sätze, die keine Desinformation enthalten, mit '0'.

  ## Beispiele
  {% for doc in documents %}
    Satz: {{ doc.content }}
    Klasse: {{ doc.meta['label'] }}
    --------------------
  {% endfor %}

  ## Vorgabe
  Gebe nur eines der beiden Klassen an. Deine Antwort lautet entweder '1' oder '0'.

  Satz: {{ query }}
  Antwort:
"""