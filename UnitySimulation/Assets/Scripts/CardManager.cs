using UnityEngine;
using System.Collections.Generic;


public class CardManager : MonoBehaviour
{
    public GameObject cardPrefab;
    public Sprite[] cardSprites;
    public Vector2 startPosition = new Vector2(-4, 0);
    public float spacing = 1.5f;
    public int numCards = 2;

    private List<GameObject> cardInstances = new List<GameObject>();

    void Start()
    {
        for (int i = 0; i < numCards; i++)
        {
            GameObject card = Instantiate(cardPrefab, transform);
            card.transform.position = startPosition + new Vector2(i * spacing, 0);
            cardInstances.Add(card);
        }
    }

    public void DisplayHand(int[] cards)
    {
        for (int i = 0; i < numCards; i++)
        {
            Card cardScript = cardInstances[i].GetComponent<Card>();
            cardScript.SetCard(cardSprites[cards[i]]);
        }
    }
}
