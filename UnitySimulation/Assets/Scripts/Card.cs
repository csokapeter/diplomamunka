using UnityEngine;

public class Card : MonoBehaviour
{
    public SpriteRenderer spriteRenderer;

    public void SetCard(Sprite cardSprite)
    {
        spriteRenderer.sprite = cardSprite;
    }
}