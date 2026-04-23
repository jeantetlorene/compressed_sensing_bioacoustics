import datetime
from copy import deepcopy
import glob, os
from yattag import Doc, indent
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import shutil
import time
import re
import io
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from preprocess import *



class Evaluation:
    def __init__(
        self,
        species_folder: str,
        settings,
        overlap=0.25,
        nb_to_group=2,
        threshold=0.5,
        step_size=1,
        method_compression=None, 
        parameter_compression=None,
        force_calc_amplitudes: bool = False,
        save_folder: str = "Predictions",
        audio_extension=".wav",

    ) -> None:
        


        self.species_folder = species_folder

        self.config = settings
        self.original_sample_rate=self.config.preprocessing.dict()["sample_rate"]
        self.segment_duration=self.config.preprocessing.dict()["segment_duration"]
        self.positive_class = self.config.data.dict()["positive_class"]
        self.negative_class = self.config.data.dict()["negative_class"]
        self.audio_extension=audio_extension
        self.overlap=overlap
        self.nb_to_group=nb_to_group
        self.threshold=threshold 
        self.step_size=step_size
        self.method_compression=method_compression
        self.parameter_compression=parameter_compression
        
        
        self.force_calc_amplitudes = force_calc_amplitudes
        self.save_amplitudes_path = Path(species_folder, "amplitudes_to_predict_" + self.audio_extension[0:])
        os.makedirs(self.save_amplitudes_path, exist_ok=True)
        
        self.save_folder = save_folder + "_" + self.audio_extension[0:]
        self.save_results = Path(self.species_folder, self.save_folder)

        self.prep = Preprocessing(
            **self.config.preprocessing.dict(),
            positive_class=self.config.data.dict()["positive_class"],
            negative_class=self.config.data.dict()["negative_class"],
            species_folder=self.species_folder,
        )

    def _group_consecutives(self, vals, step=1):
        """Return list of consecutive lists of numbers from vals (number list)."""
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    def _group(self, L):
        L.sort()
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last:  # Part of the group, bump the end
                last = n
            else:  # Not part of the group, yield current group and start a new
                yield first, last
                first = last = n
        yield first, last  # Yield the last group

    def _dataframe_to_svl(self, dataframe, sample_rate, length_audio_file_frames):
        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        doc.asis("<!DOCTYPE sonic-visualiser>")

        with tag("sv"):
            with tag("data"):
                model_string = '<model id="1" name="" sampleRate="{}" start="0" end="{}" type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" subtype="box" minimum="0" maximum="{}" units="Hz" />'.format(
                    sample_rate, length_audio_file_frames, sample_rate / 2
                )
                doc.asis(model_string)

                with tag("dataset", id="0", dimensions="2"):
                    # Read dataframe or other data structure and add the values here
                    # These are added as "point" elements, for example:
                    # '<point frame="15360" value="3136.87" duration="1724416" extent="2139.22" label="Cape Robin" />'
                    for index, row in dataframe.iterrows():
                        point = '<point frame="{}" value="{}" duration="{}" extent="{}" label="{}" />'.format(
                            int(int(row["start(sec)"]) * sample_rate),
                            int(row["low(freq)"]),
                            int(
                                (int(row["end(sec)"]) - int(row["start(sec)"]))
                                * sample_rate
                            ),
                            int(row["high(freq)"]),
                            row["label"],
                        )

                        # add the point
                        doc.asis(point)
            with tag("display"):
                display_string = '<layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
                doc.asis(display_string)

        result = indent(doc.getvalue(), indentation=" " * 2, newline="\r\n")

        return result


    def _predictions(self, model, inputs, batch_size):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prediction_list = []
        model = model.to(device)
        model.eval()
        #np.save("inputs.npy", inputs)
        X_tensor = torch.from_numpy(inputs).float()

        # Reshape X_tensor
        # print("X_TENSOR SHAPE", X_tensor.shape)
        if len(X_tensor.shape) == 3:
            X_tensor = X_tensor.unsqueeze(1)

        loader = torch.utils.data.DataLoader(
            X_tensor, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                prediction_list.append(pred.cpu())

        softmax_prediction = [i.detach().numpy() for i in prediction_list]
        return np.vstack(softmax_prediction)

    def _calc_amplitudes_to_predict(self, file_name_no_extension, preprocessing_arg, Saved_amplitudes=True,  verbose = True):
        if str(
            Path(self.prep.audio_path, file_name_no_extension + self.audio_extension)
        ) in glob(str(self.prep.audio_path / f"*{self.audio_extension}")):
            print("Found file")
        
        init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        print(f"{init_time} Load {file_name_no_extension} data ")


        # Read audio file
        if self.audio_extension==".npy":
                original_sample_rate=self.original_sample_rate
                #Construct the folder path
                folder_path = Path(self.species_folder) / "Compressed_Audio" / f"{self.method_compression}_reconstructed_{self.parameter_compression}"

                # Search for the file using glob
                matching_files = list(folder_path.glob(f"{file_name_no_extension}_*.npy"))
                # Filter to select the one ending with the exact suffix
                pattern = re.compile(re.escape(file_name_no_extension) + r"(_\d+)?_reconstructed\.npy$")

                # Filter files with the correct ending
                filtered_files = [f for f in matching_files if pattern.fullmatch(f.name)]

                if filtered_files:
                    file_path = filtered_files[0]
                    audio_amps = np.load(file_path)
                else:
                    raise FileNotFoundError(f"No file found for pattern: {file_name_no_extension}_*.npy in {folder_path}")
        else : 
            audio_amps, original_sample_rate = self.prep.read_audio_file(file_name_no_extension, self.method_compression, self.parameter_compression)

        
        if preprocessing_arg==True : 
            print("Filtering...") if verbose else None
            filtered = self.prep.butter_lowpass_filter(
                audio_amps, self.prep.lowpass_cutoff, self.prep.nyquist_rate
                )

            print("Downsampling...") if verbose else None 
            amplitudes, sample_rate = self.prep.downsample_file(
                filtered, original_sample_rate, self.prep.downsample_rate
                )
            del filtered  
        else : 
            amplitudes = audio_amps
            sample_rate= original_sample_rate

        len_audio_amps=len(audio_amps)
        del audio_amps

        start_values = np.arange(
            0, len(amplitudes) / sample_rate - self.prep.segment_duration, self.step_size
        ).astype(int)
        end_values = start_values + self.prep.segment_duration


        amplitudes_to_predict = []

        init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        print(f"{init_time} Create list of spectrograms")

        for s, e in zip(start_values, end_values):
            S = self.prep.convert_single_to_image(
                amplitudes[s * sample_rate : e * sample_rate], sample_rate
            )
            amplitudes_to_predict.append(S)


        del amplitudes 
        amplitudes_to_predict = np.asarray(amplitudes_to_predict)
        
        
        # save to disk
        if Saved_amplitudes : 

            save_dict = {
                "amplitudes_to_predict": amplitudes_to_predict,
                "sample_rate": sample_rate,
                "original_sample_rate": original_sample_rate,
                "len_audio_amps": len_audio_amps,
                }
            # Save to results folder
            save_name = Path(self.save_amplitudes_path, file_name_no_extension + "_amplitudes_to_predict.npy")
            np.save(save_name, save_dict)
            print("Saved amplitudes to predict to disk: ", save_name)
        
        
        return  amplitudes_to_predict, sample_rate, original_sample_rate, len_audio_amps
    
    
    def _get_amplitudes_to_predict(self, file_name_no_extension, preprocessing_arg, verbose = True):
        # Change this to True to force recalculation of amplitudes to predict, make this a parameter later
        if self.force_calc_amplitudes:
            print("Forcing recalculation of amplitudes to predict")
            return self._calc_amplitudes_to_predict(file_name_no_extension, preprocessing_arg, Saved_amplitudes=False, verbose=verbose)
        # Check if the amplitudes to predict have already been calculated
        else : 
            save_name = Path(self.save_amplitudes_path, file_name_no_extension + "_amplitudes_to_predict.npy")
            if os.path.exists(save_name):
                print("Found amplitudes to predict on disk: ", save_name)
                data = np.load(save_name, allow_pickle=True)
                amplitudes_to_predict = data.item().get("amplitudes_to_predict")
                sample_rate = data.item().get("sample_rate")
                original_sample_rate = data.item().get("original_sample_rate")
                len_audio_amps = data.item().get("len_audio_amps")
                return amplitudes_to_predict, sample_rate, original_sample_rate, len_audio_amps
            else:
                print("No amplitudes to predict found on disk")
                return self._calc_amplitudes_to_predict(file_name_no_extension, preprocessing_arg, Saved_amplitudes=True, verbose=verbose)
            
    # Merge overlapping or adjacent intervals
    def _merge_intervals(self, intervals, gap=0.0):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged
    
    def _process_one_file(self, file_name_no_extension, model, preprocessing_arg, verbose=True):
                       

        amplitudes_to_predict, sample_rate, original_sample_rate, len_audio_amps = self._get_amplitudes_to_predict(file_name_no_extension, preprocessing_arg, verbose=verbose)
        
     
        print("Predicting...")
        
        # predictions
        softmax_predictions = self._predictions(
            model, amplitudes_to_predict, batch_size=128
        )
        
        # After softmax_predictions = self._predictions(...)

        del amplitudes_to_predict

        start_times = np.arange(0,
        len(softmax_predictions) * self.step_size, self.step_size)
        # Convert predictions to binary with timestamps
        positive_intervals = []
        for i, softmax_values in enumerate(softmax_predictions):
            if softmax_values[1] >= self.threshold:
                start = start_times[i]
                end = start + self.prep.segment_duration
                positive_intervals.append((start, end))

        merged = self._merge_intervals(positive_intervals, gap=0.0)

        # Filter by minimum duration (replaces nb_to_group)
        min_duration = self.nb_to_group * self.step_size  # keep same logic
        merged = [(s, e) for s, e in merged if (e - s) >= min_duration]


        if len(merged) > 0:
            # Create a dataframe to store each prediction
            df_values = []
            prediction_name = "predicted_baseline"
            for start, end in merged:
                df_values.append([
                       start,
                        end,
                        600,
                        2000,
                        prediction_name,
                    ])
            df_preds = pd.DataFrame(
                df_values,
                columns=[["start(sec)", "end(sec)", "low(freq)", "high(freq)", "label"]],
            )

            # Create a .svl outpupt file
            xml = self._dataframe_to_svl(
                df_preds, original_sample_rate, len_audio_amps
            )

            # Write the .svl file
            # text_file = open(species_folder+'/Model_Output/'+file_name_no_extension+"_"+self.model_type+".svl", "w")
            text_file = open(
                "{}_predictions.svl".format(
                    str(Path(self.save_results, file_name_no_extension))
                ),
                "w",
            )
            n = text_file.write(xml)
            text_file.close()

        else:
            print("No detected calls to save.")
        #del amplitudes

        print("Done")

    def prediction_files(self, model, type, preprocessing_arg):
        
        
        test_path = Path(self.species_folder, "DataFiles", type + ".txt")
        #test_path = Path(self.species_folder, "DataFiles", "test.txt")
        file_names = pd.read_csv(test_path, header=None)


        if os.path.isdir(self.save_results) == False:
            print("creating the folder")
        else:
            shutil.rmtree(self.save_results)
            print("clean the folder")

        os.mkdir(self.save_results)

        for file in file_names.values:
            file_name_no_extension = file[0]

            print("Processing file:", file)
            init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            print(f"{init_time} Processing file: {file} ")

            self._process_one_file(file_name_no_extension, model, preprocessing_arg, verbose=True)

    def _overlap(self, start1, end1, start2, end2):
        """how much does the range (start1, end1) overlap with (start2, end2)"""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        return max(0, overlap_end - overlap_start)

    def _repair_svl(
        self, file_names, annotation_folder
    ):
        saved_folder = Path(self.species_folder, "Annotations_corrected")
        os.makedirs(saved_folder, exist_ok=True)

        for file in file_names.values:
            file = file[0]
            reader = AnnotationReader(
                self.species_folder, file, self.prep.annotation_extension, self.audio_extension, self.positive_class,
            )

            audio_amps, original_sample_rate = self.prep.read_audio_file(file, self.method_compression, self.parameter_compression)
            

            (
                df,
                sampleRate,
                start_m,
                end_m,
            ) = reader.get_annotation_information_testing(annotation_folder, self.prep.annotation_extension)

            new_frames = []
            new_values = []
            new_extents = []
            new_durations = []
            new_labels = []

            for i in range(0, int(len(audio_amps) / original_sample_rate), self.segment_duration ):
                index_start = i
                index_end = i + self.segment_duration

                overlap_label = False

                for index, row in df.iterrows():
                    labeled_start = row["frame"] / int(sampleRate)
                    labeled_end = (row["frame"] + row["duration"]) / int(sampleRate)
                    
                    if self._overlap(index_start, index_end , labeled_start, labeled_end) > 0 :
                        if row["label"] == self.positive_class :
                            overlap_label = True

                    """
                    if index_start >= labeled_start and index_start <= labeled_end:
                        if row["label"] == self.positive_class:
                            overlap_label = True

                    if index_end >= labeled_start and index_end <= labeled_end:
                        if row["label"] == self.positive_class:
                            overlap_label = True
                    """

                if overlap_label != True:
                    new_frames.append(index_start * int(sampleRate))
                    new_values.append(700)
                    new_durations.append(self.segment_duration * int(sampleRate))
                    new_extents.append(1500)
                    new_labels.append(self.negative_class)

            df_repaired = pd.DataFrame(
                {
                    "frame": new_frames,
                    "value": new_values,
                    "duration": new_durations,
                    "extent": new_extents,
                    "label": new_labels,
                }
            )

            df_repaired = pd.concat(
                [df_repaired, df[df["label"] == self.positive_class]]
            )

            xml = reader.dataframe_to_svl(df_repaired, sampleRate, start_m, end_m)
            saved_folder_path = Path(saved_folder)
            text_file_path = saved_folder_path / f"{file}_repaired.svl"
            text_file = open(str(text_file_path), "a")
            n = text_file.write(xml)
            text_file.close()

        

    def comparison_predictions_annotations(self, folder, type="test"):
        
        print("comparing prediction and annotation")
        test_path = Path(self.species_folder, "DataFiles", type + ".txt")
        file_names = pd.read_csv(test_path, header=None)



        predictions = []
        annotations = []

        # check if corrected annotations for the testing files have been done
        if os.path.exists(Path(self.species_folder, "Annotations_corrected")) and len(os.listdir(Path(self.species_folder, "Annotations_corrected")))>0:
            print(
                "the corrected annotations of the testing dataset have already been created "
            )

        else:
            print(
                "Need to modify the annotations of the testing dataset to allow a correct evaluation of the model "
            )
            self._repair_svl(
                file_names,
                annotation_folder="Annotations",
            )
        for file in file_names.values:
            file = file[0]

            
            reader = AnnotationReader(
                self.species_folder,
                file,
                annotation_extension=self.prep.annotation_extension,
                audio_extension=self.audio_extension,
                positive_class=self.positive_class,
            )

            svl = reader.get_annotation_information(annotation_folder="Annotations_corrected",sufix_file="_repaired.svl")[0]
            svl["Overlap"] = 0.0
            svl["Cat"] = "TN"
            svl.loc[svl.Label == self.positive_class, "Cat"] = "FN"
            svl["Index"] = np.nan
            svl["Nb overlap"] = 0
            svl["Name"] = file

            if os.path.exists(
                Path(self.species_folder, folder, file + "_predictions.svl")
            ):
                print("Found Prediction: ", file)
                predict = reader.get_annotation_information(
                    annotation_folder=folder, sufix_file="_predictions.svl")[0]

                predict["Overlap"] = 0.0
                predict["Cat"] = "FP"
                predict["Index"] = np.nan
                predict["Nb overlap"] = 0
                predict["Name"] = file

                # compare predictions vs annotations
                if svl[svl.Label == self.positive_class].shape[0] != 0:
                    for index, row in predict.iterrows():
                        idx = np.abs(
                            np.asarray(
                                svl[svl.Label == self.positive_class]["Start"]
                            )
                            - row.iloc[0]
                        ).argmin()  # get the closest window
                        lap = self._overlap(
                            row.iloc[0],
                            row.iloc[1],
                            svl[svl.Label == self.positive_class].iloc[idx, 0],
                            svl[svl.Label == self.positive_class].iloc[idx, 1],
                        )  # check overlap

                        if lap > self.overlap * self.segment_duration :
                            predict.loc[index, "Overlap"] = deepcopy(lap)
                            predict.loc[index, "Cat"] = "TP"
                            predict.loc[index, "Index"] = idx
                        else:
                            predict.loc[index, "Overlap"] = deepcopy(lap)

                    for index, row in predict.iterrows():
                        w = 0
                        for idx_svl, row_svl in svl[
                            svl.Label == self.positive_class
                        ].iterrows():
                            lap = self._overlap(
                                row.iloc[0],
                                row.iloc[1],
                                row_svl.iloc[0],
                                row_svl.iloc[1],
                            )
                            if lap > self.overlap * self.segment_duration :
                                w += 1
                        predict.loc[index, "Nb overlap"] = w
                else:
                    print("No positive class in the annotation file")
                predictions.append(predict)

                # compare annotations vs predictions
                for index, row in svl.iterrows():
                    idx = np.abs(
                        np.asarray(predict["Start"]) - row.iloc[0]
                    ).argmin()  # get the closest window
                    lap = self._overlap(
                        row.iloc[0],
                        row.iloc[1],
                        predict.iloc[idx, 0],
                        predict.iloc[idx, 1],
                    )  # check overlap

                    if (lap > self.overlap * self.segment_duration) & (
                        svl.loc[index, "Label"] == self.positive_class
                    ):
                        svl.loc[index, "Overlap"] = deepcopy(lap)
                        svl.loc[index, "Index"] = idx
                        svl.loc[index, "Cat"] = "TP"
                    elif (lap > self.overlap * self.segment_duration) & (
                        svl.loc[index, "Label"] == self.negative_class
                    ):
                        svl.loc[index, "Overlap"] = deepcopy(lap)
                        svl.loc[index, "Index"] = idx
                        svl.loc[index, "Cat"] = "FP"
                    else:
                        svl.loc[index, "Overlap"] = deepcopy(lap)

                # Print File and FP TP FN
                print("-------------")
                print(file)
                print("FP : ", predict[predict.Cat == "FP"].shape[0])
                print("TP : ", svl[svl.Cat == "TP"].shape[0])
                print("FN : ", svl[svl.Cat == "FN"].shape[0])
                print("-------------")

                for index, row in svl.iterrows():
                    w = 0
                    for idx_pred, row_pred in predict.iterrows():
                        lap = self._overlap(
                            row.iloc[0], row.iloc[1], row_pred.iloc[0], row_pred.iloc[1]
                        )
                        if lap > self.overlap * self.segment_duration:
                            w += 1
                    svl.loc[index, "Nb overlap"] = w

            annotations.append(svl)

        if predictions : 
            Predictions = pd.DataFrame(np.concatenate(predictions, axis=0))
            Predictions.columns = predict.columns
            Predictions.Index = Predictions.Index.astype(float)
        else : 
            Predictions=pd.DataFrame()

        Annotations = pd.DataFrame(np.concatenate(annotations, axis=0))
        Annotations.columns = svl.columns
        Annotations.Index = Annotations.Index.astype(float)

        return Predictions, Annotations

    def testing_score(self, Annotations, Predictions):

        if not Predictions.empty:
            cat, count = np.unique(Predictions["Cat"], return_counts=True)
            print("Predictions:", dict(zip(cat, count)))
        else:
            cat = np.array(["FP"])
            count = np.array([0])

        cat_a, count_a = np.unique(Annotations["Cat"], return_counts=True)
        print("Annotations:", dict(zip(cat_a, count_a)))


        pred_counts = dict(zip(cat, count))
        anno_counts = dict(zip(cat_a, count_a))

        FP = pred_counts.get("FP", 0)
        TP = anno_counts.get("TP", 0)
        FN = anno_counts.get("FN", 0)
        TN = anno_counts.get("TN", 0)

        F_score = TP / (TP + (FN + FP) / 2) if (TP + (FN + FP) / 2) > 0 else 0
        Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        confusion=np.array([[TP, FP], [FN, TN]])
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(
            "Number of calls to detect : ",
            Annotations[Annotations.Label == self.positive_class].shape[0],
        )
        print()
        print("False Positif : ", FP)
        print("True Positif : ", TP)
        print("False Negatif : ", FN)
        print()
        print("F1-score : ", F_score)
        print("Accuracy : ", Accuracy)

        return F_score, Accuracy, confusion, precision, recall


    def _testing_dataset_run(self, model, type="test", print_report=True):
        data_path = Path(self.species_folder, type)
        
        if self.method_compression!= None : 
            with open(Path(data_path, self.positive_class +"_X_" + type + "_" + self.method_compression + "_"+ self.parameter_compression + ".pkl"), "rb") as f:
                X = pickle.load(f)
        else : 
            with open(Path(data_path, self.positive_class + "_X_" + type + ".pkl"), "rb") as f:
                X = pickle.load(f)
        
        with open(Path(data_path, self.positive_class + "_Y_" + type + ".pkl"), "rb") as f:
            Y = pickle.load(f)

        print("Data Loaded from: ", data_path)
        print("Evaluating...")
        
        starting = time.time()
        predictions = self._predictions(model, X, batch_size=128)
        execution_time = time.time() - starting

        if self.threshold is None:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions[:,1] > self.threshold).astype(int)
        
        targets = np.argmax(Y, axis=1)
        
      
        f1 = f1_score(targets, predictions)
        report = classification_report(targets, predictions)
        confusion = confusion_matrix(targets, predictions)
        if print_report:
            print("f1 score :", f1)
            print(report)
            print(confusion)
         
            return f1, confusion

    def _entire_files_run(self, model, type, preprocessing_arg):
    
        starting = time.time()
        self.prediction_files(model, type, preprocessing_arg)
        execution_time = time.time() - starting
        
        Predictions, Annotations = self.comparison_predictions_annotations(
            self.save_folder, type=type
        )

        F_score, Accuracy, confusion, precision, recall  = self.testing_score(Annotations, Predictions)
        
        return F_score, confusion, execution_time, precision, recall
    
    
    def run(self, model, type="test", test_type = "testing_dataset", preprocessing_arg=True):
        if test_type == "testing_dataset":
            return self._testing_dataset_run(model, type=type )
        
        else : 
            return self._entire_files_run(model, type=type, preprocessing_arg=preprocessing_arg)
        
        
