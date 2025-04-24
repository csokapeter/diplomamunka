using UnityEngine;
using UnityEngine.UI;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using TMPro;

public class EnvController : Agent
{
    public GameObject community;
    public GameObject player1;
    public GameObject player2;
    public GameObject agent;

    private CardManager communityCM;
    private CardManager player1CM;
    private CardManager player2CM;
    private CardManager agentCM;

    public Button myButton;
    private bool nextStep = false;

    public TMP_Text pot;
    public TMP_Text sidePot;
    public TMP_Text agentLastAction;
    public TMP_Text player1LastAction;
    public TMP_Text player2LastAction;
    public TMP_Text agentChips;
    public TMP_Text player1Chips;
    public TMP_Text player2Chips;
    public TMP_Text agentPosition;
    public TMP_Text player1Position;
    public TMP_Text player2Position;

    private static readonly string[] actionNames = {
        "-",       // -1
        "Fold",    // 0
        "Check",   // 1
        "Call",    // 2
        "Raise",   // 3
        "All-in"   // 4
    };

    private static readonly string[] positionNames = {
        "Dealer",       // 0
        "Small Blind",  // 1
        "Big Blind"     // 2
    };

    void Start()
    {
        myButton.onClick.AddListener(() => {
            nextStep = true;
        });
        communityCM = community.GetComponent<CardManager>();
        player1CM = player1.GetComponent<CardManager>();
        player2CM = player2.GetComponent<CardManager>();
        agentCM = agent.GetComponent<CardManager>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (nextStep)
        {
            sensor.AddObservation(1);
            nextStep = false;
        }
        else
        {
            sensor.AddObservation(0);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        bool allZero = true;
        for (int i = 0; i < actions.ContinuousActions.Length; i++)
        {
            if (actions.ContinuousActions[i] != 0f)
            {
                allZero = false;
                break;
            }
        }

        if (!allZero)
        {
            int[] communityCards = new int[5] {
                (Mathf.Max(0, (int)actions.ContinuousActions[0] / 100 - 1)) * 13 + (int)actions.ContinuousActions[0] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[1] / 100 - 1)) * 13 + (int)actions.ContinuousActions[1] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[2] / 100 - 1)) * 13 + (int)actions.ContinuousActions[2] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[3] / 100 - 1)) * 13 + (int)actions.ContinuousActions[3] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[4] / 100 - 1)) * 13 + (int)actions.ContinuousActions[4] % 100};
            communityCM.DisplayHand(communityCards);
            int[] agentCards = new int[2] {
                (Mathf.Max(0, (int)actions.ContinuousActions[5] / 100 - 1)) * 13 + (int)actions.ContinuousActions[5] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[6] / 100 - 1)) * 13 + (int)actions.ContinuousActions[6] % 100};
            agentCM.DisplayHand(agentCards);
            int[] player1Cards = new int[2] {
                (Mathf.Max(0, (int)actions.ContinuousActions[7] / 100 - 1)) * 13 + (int)actions.ContinuousActions[7] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[8] / 100 - 1)) * 13 + (int)actions.ContinuousActions[8] % 100};
            player1CM.DisplayHand(player1Cards);
            int[] player2Cards = new int[2] {
                (Mathf.Max(0, (int)actions.ContinuousActions[9] / 100 - 1)) * 13 + (int)actions.ContinuousActions[9] % 100, 
                (Mathf.Max(0, (int)actions.ContinuousActions[10] / 100 - 1)) * 13 + (int)actions.ContinuousActions[10] % 100};
            player2CM.DisplayHand(player2Cards);

            agentLastAction.text = $"Last action: {actionNames[(int)actions.ContinuousActions[11] + 1]}";
            player1LastAction.text = $"Last action: {actionNames[(int)actions.ContinuousActions[12] + 1]}";
            player2LastAction.text = $"Last action: {actionNames[(int)actions.ContinuousActions[13] + 1]}";

            agentPosition.text = $"Position: {positionNames[(int)actions.ContinuousActions[14]]}";
            player1Position.text = $"Position: {positionNames[(int)actions.ContinuousActions[15]]}";
            player2Position.text = $"Position: {positionNames[(int)actions.ContinuousActions[16]]}";

            agentChips.text = $"Chips: {(int)actions.ContinuousActions[17]}";
            player1Chips.text = $"Chips: {(int)actions.ContinuousActions[18]}";
            player2Chips.text = $"Chips: {(int)actions.ContinuousActions[19]}";

            if ((int)actions.ContinuousActions[20] == 1)
                agentLastAction.text = $"Last action: {actionNames[1]}";
            if ((int)actions.ContinuousActions[21] == 1)
                player1LastAction.text = $"Last action: {actionNames[1]}";
            if ((int)actions.ContinuousActions[22] == 1)
                player2LastAction.text = $"Last action: {actionNames[1]}";

            pot.text = $"Pot: {(int)actions.ContinuousActions[23]}";
            sidePot.text = $"Side pot: {(int)actions.ContinuousActions[24]}";
        }
    }
}
