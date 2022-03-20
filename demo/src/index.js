import * as tf from '@tensorflow/tfjs';

import {Hands, HAND_CONNECTIONS} from "@mediapipe/hands";
import {Camera} from "@mediapipe/camera_utils";
import {drawConnectors, drawLandmarks} from "@mediapipe/drawing_utils";

import Splitting from "splitting";
import {getRandomWord} from './wordlist';


const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
const aslModel = await tf.loadLayersModel('asl_model/model.json');

let DRAW_LANDMARKS = false;
document.getElementById('checkboxDrawLandmarks').onclick = () => {
    DRAW_LANDMARKS = document.getElementById('checkboxDrawLandmarks').checked
};

const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let wordElement, spellingProgress, currentWord, wordChars;
randomizeWord();

videoElement.addEventListener('play', (event) => {
    let [videoWidth, videoHeight] = [event.target.videoWidth, event.target.videoHeight];
    canvasElement.setAttribute("width", videoWidth.toString() + "px");
    canvasElement.setAttribute("height", videoHeight.toString() + "px");
});

let [letter, suggestedLetter] = ["", ""];
setInterval(function() {
    // set interval, to buffer the predicted letter changes.
    if (suggestedLetter !== letter) {
        letter = suggestedLetter;
    }
}, 500)

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            if (DRAW_LANDMARKS) {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5});
                drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
            }

            let isRight = results.multiHandedness[0].label === "Right";
            let prediction = aslModel.predict(preprocessLandmarks(landmarks, isRight));
            suggestedLetter = ALPHABET[prediction.argMax(1).bufferSync().get(0)];

            // Draw the letter in the canvas.
            canvasCtx.font = '50px Sans-serif';
            canvasCtx.strokeStyle = 'black';
            canvasCtx.lineWidth = 8;
            canvasCtx.strokeText(letter,
                landmarks[0].x * canvasCtx.canvas.width,
                (landmarks[0].y + 0.1) * canvasCtx.canvas.height);
            canvasCtx.fillStyle = 'white';
            canvasCtx.fillText(letter,
                landmarks[0].x * canvasCtx.canvas.width,
                (landmarks[0].y + 0.1) * canvasCtx.canvas.height);

            if (letter === currentWord[spellingProgress.length].toUpperCase()) {
                // Mark the correctly spelled letter red.
                wordChars.chars[spellingProgress.length].setAttribute("class", "text-danger");
                spellingProgress += letter;

                // If the word is fully spelled, get a new random word.
                if (currentWord.toUpperCase() === spellingProgress) {
                    spellingProgress = "";
                    randomizeWord();
                }
            }
        }
    }

    canvasCtx.restore();
}

function preprocessLandmarks(landmarks, isRight) {
    let result = [];
    for (let i = 0; i <= 20; i++) {
        result.push(isRight ? landmarks[i].x : 1 - landmarks[i].x); // Flip the x-axis, if the hand is recognized as left hand.
        result.push(landmarks[i].y);
        result.push(landmarks[i].z);
    }

    return tf.tensor([result]);
}

function randomizeWord() {
    // Remove the span containing the old word, and create a new span for the new word.
    document.getElementById('word').firstElementChild.remove();
    wordElement = document.createElement("span");
    document.getElementById('word').append(wordElement)

    // Reset the spelling progress.
    spellingProgress = "";

    // Randomize the current word and split into single characters.
    currentWord = getRandomWord();
    wordElement.innerText = currentWord;
    wordChars = Splitting({target: wordElement, by: 'chars'})[0];

    // Add a tooltip for every letter, containing an image of the correct ASL sign.
    wordChars.chars.forEach(function (char){
        char.setAttribute("data-bs-toggle", "tooltip");
        char.setAttribute("data-bs-placement", "bottom");
        char.setAttribute("data-bs-html", "true");
        char.setAttribute("title", "<span class='asltext'>" + char.dataset.char + "</span>");
    });

    // Activtate the Bootstrap tooltips.
    let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
}

const hands = new Hands({
    locateFile: (file) => {
        return `mediapipe_hands/${file}`;
    }
});
hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    }
});
camera.start();
