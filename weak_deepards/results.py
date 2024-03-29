import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy import interp
from sklearn.metrics import auc, roc_curve

#from metrics import janky_roc


class PatientResults(object):
    def __init__(self, patient_id, ground_truth, fold_idx, epoch):
        self.patient_id = patient_id
        self.ground_truth = ground_truth
        self.majority_prediction = np.nan
        self.fold_idx = fold_idx
        self.epoch = epoch
        self.predictions = None
        self.ards_votes = 0
        self.other_votes = 0

    def set_results(self, predictions):
        self.predictions = predictions
        # XXX make sure that you format predictions into something this can handle
        maj_pred = [1 if pred > .5 else 0 for pred in self.predictions]
        self.majority_prediction = 1 if sum(maj_pred) / len(maj_pred) > .5 else 0
        self.ards_votes = sum(maj_pred)
        self.other_votes = len(maj_pred) - self.ards_votes

    def to_list(self):
        return [
            self.patient_id,
            self.other_votes,
            self.ards_votes,
            self.ards_votes / (self.other_votes + self.ards_votes),
            self.majority_prediction,
            self.fold_idx,
            self.epoch,
            self.ground_truth,
        ], ['patient_id', 'other_votes', 'ards_votes', 'frac_votes', 'majority_prediction', 'fold_idx', 'epoch', 'ground_truth']


class ModelResults(object):
    def __init__(self, fold_idx, epoch):
        self.fold_idx = fold_idx
        self.all_patient_results = []
        self.epoch = epoch

    def set_results(self, y_test, predictions):
        """
        Save model results

        :param y_test: dataframe for y_test ground truth
        :param predictions: predictions made for the patient
        """
        for pt, pt_rows in y_test.groupby('patient'):
            pt_predictions = predictions.loc[pt_rows.index]
            ground_truth_label = pt_rows.iloc[0].y
            results = PatientResults(pt, ground_truth_label, self.fold_idx, self.epoch)
            results.set_results(pt_predictions)
            self.all_patient_results.append(results)

    def get_patient_results_dataframe(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def get_patient_results(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.to_list()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def get_patient_hourly_preds(self):
        tmp = []
        for result in self.all_patient_results:
            lst, cols = result.get_hourly_preds()
            tmp.append(lst)
        return pd.DataFrame(tmp, columns=cols)

    def count_predictions(self):
        """
        """
        threshold = .5
        stat_cols = []
        for patho in ['other', 'ards']:
            stat_cols.extend([
                '{}_tps_{}'.format(patho, threshold),
                '{}_tns_{}'.format(patho, threshold),
                '{}_fps_{}'.format(patho, threshold),
                '{}_fns_{}'.format(patho, threshold)
            ])
        stat_cols += ['fold_idx', 'epoch']

        patient_results = self.get_patient_results()
        stat_results = []
        for patho in [0, 1]:
            # The 2 idx is the prediction fraction from the patient results class
            #
            # In this if statement we are differentiating between predictions made
            # for ARDS and predictions made otherwise. the eq_mask signifies
            # predictions made for the pathophysiology. For instance if our pathophys
            # is 0 then we want the fraction votes for ARDS to be < prediction threshold.
            if patho == 0:
                eq_mask = patient_results.frac_votes < threshold
                neq_mask = patient_results.frac_votes >= threshold
            else:
                eq_mask = patient_results.frac_votes >= threshold
                neq_mask = patient_results.frac_votes < threshold

            stat_results.extend([
                len(patient_results[eq_mask][patient_results.loc[eq_mask, 'ground_truth'] == patho]),
                len(patient_results[neq_mask][patient_results.loc[neq_mask, 'ground_truth'] != patho]),
                len(patient_results[eq_mask][patient_results.loc[eq_mask, 'ground_truth'] != patho]),
                len(patient_results[neq_mask][patient_results.loc[neq_mask, 'ground_truth'] == patho]),
            ])
        return stat_results + [self.fold_idx, self.epoch], stat_cols


class ModelCollection(object):
    def __init__(self):
        self.models = []
        self.model_results = {
            'folds': {},
            'epoch': {},
            'aggregate': None,
        }

    def add_model(self, y_test, predictions, fold_idx, epoch):
        # sanity check
        #assert len(y_test.dropna()) == len(y_test)
        model = ModelResults(fold_idx, epoch)
        model.set_results(y_test, predictions)
        self.models.append(model)

    def get_aggregate_predictions_dataframe(self):
        """
        Get aggregated results of all the dataframes
        """
        tmp = []
        for model in self.models:
            results, cols = model.count_predictions()
            tmp.append(results)
        return pd.DataFrame(tmp, columns=cols)

    #def get_all_hourly_preds(self):
    #    tmp = [model.get_patient_hourly_preds() for model in self.models]
    #    return pd.concat(tmp, ignore_index=True)

    def get_all_patient_results_dataframe(self):
        tmp = [model.get_patient_results_dataframe() for model in self.models]
        return pd.concat(tmp, axis=0, ignore_index=True)

    def get_all_patient_results_in_fold_dataframe(self, fold_idx):
        # if you don't want to reconstitute this all the time you
        # can probably keep a boolean variable that tells you when you need to remake
        # and then can store as a global var
        tmp = [model.get_patient_results_dataframe() for model in self.models if model.fold_idx == fold_idx]
        return pd.concat(tmp, axis=0, ignore_index=True)

    def get_all_patient_results_in_epoch_dataframe(self, epoch):
        tmp = [model.get_patient_results_dataframe() for model in self.models if model.epoch == epoch]
        return pd.concat(tmp, axis=0, ignore_index=True)

    def calc_epoch_stats(self, epoch, print_results=True):
        df = self.get_aggregate_predictions_dataframe()
        epoch_results = df[df.epoch == epoch]
        patient_results = self.get_all_patient_results_in_epoch_dataframe(epoch)
        results_df = self.calc_results(epoch_results, patient_results)
        self.model_results['epoch'][epoch] = results_df
        if print_results:
            self.print_results_table(results_df)

    def calc_fold_stats(self, fold_idx, print_results=True):
        df = self.get_aggregate_predictions_dataframe()
        fold_results = df[df.fold_idx == fold_idx]
        patient_results = self.get_all_patient_results_in_fold_dataframe(fold_idx)
        results_df = self.calc_results(fold_results, patient_results)
        self.model_results['folds'][fold_idx] = results_df
        if print_results:
            self.print_results_table(results_df)

    def calc_aggregate_stats(self, print_results=True):
        df = self.get_aggregate_predictions_dataframe()
        patient_results = self.get_all_patient_results_dataframe()
        results_df = self.calc_results(df, patient_results)
        self.model_results['aggregate'] = results_df
        if print_results:
            print('---Aggregate Results---')
            self.print_results_table(results_df)

    def calc_results(self, dataframe, patient_results):
        columns = ['epoch', 'patho', 'acc', 'recall', 'spec', 'prec', 'npv', 'auc', 'acc_ci', 'recall_ci', 'spec_ci', 'prec_ci', 'npv_ci', 'auc_ci']
        stats_tmp = []

        for epoch, epoch_results in patient_results.groupby('epoch'):
            aucs = self.get_auc_results(epoch_results)
            uniq_pts = len(epoch_results.patient_id.unique())
            mean_auc = aucs.mean().round(3)
            auc_ci = (1.96 * np.sqrt(mean_auc * (1-mean_auc) / uniq_pts)).round(3)
            for patho in ['other', 'ards']:
                stats = self.get_summary_statistics_from_frame(dataframe, patho)
                means = stats.mean().round(3)
                cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
                stats_tmp.append([
                    epoch,
                    patho,
                    means[0],
                    means[1],
                    means[2],
                    means[3],
                    means[4],
                    aucs.mean().round(2),
                    cis[0],
                    cis[1],
                    cis[2],
                    cis[3],
                    cis[4],
                    auc_ci,
                ])
        return pd.DataFrame(stats_tmp, columns=columns)

    def print_results_table(self, results_df):
        table = PrettyTable()
        table.field_names = ['epoch', 'patho', 'sensitivity', 'specificity', 'precision', 'npv', 'auc']
        for i, row in results_df.iterrows():
            results_row = [
                row.epoch,
                row.patho,
                u"{}\u00B1{}".format(row.recall, row.recall_ci),
                u"{}\u00B1{}".format(row.spec, row.spec_ci),
                u"{}\u00B1{}".format(row.prec, row.prec_ci),
                u"{}\u00B1{}".format(row.npv, row.npv_ci),
                u"{}\u00B1{}".format(row.auc, row.auc_ci),
            ]
            table.add_row(results_row)
        print(table)

    def plot_roc_all_folds(self):
        # I might be able to find confidence std using p(1-p). Nah. we actually cant do
        # this because polling is using the identity of std from a binomial distribution. So
        # in order to have conf interval we need some kind of observable std.
        tprs = []
        aucs = []
        threshes = set()
        mean_fpr = np.linspace(0, 1, 100)
        results = self.get_all_patient_results_dataframe()
        uniq_pts = len(results.patient_id.unique())

        for fold_idx in results.fold_idx.unique():
            fold_preds = results[results.fold_idx == fold_idx]
            model_aucs = self.get_auc_results(fold_preds)
            fpr, tpr, thresh = roc_curve(fold_preds.ground_truth, fold_preds.frac_votes)
            threshes.update(thresh)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (fold_idx+1, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = round(auc(mean_fpr, mean_tpr), 2)
        std_auc = np.std(aucs)

        model_aucs = self.get_auc_results(results)
        auc_ci = (1.96 * np.sqrt(mean_auc * (1-mean_auc) / uniq_pts)).round(3)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, auc_ci),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    def plot_sen_spec_vs_thresh(self, thresh_interval):
        y1 = []
        y2 = []
        pred_threshes = range(0, 100+thresh_interval, thresh_interval)
        for i in pred_threshes:
            thresh = i / 100.0
            df = self.get_aggregate_predictions_dataframe(thresh)
            stats = self.get_summary_statistics_from_frame(df, 'ards', thresh)
            means = stats.mean()
            y1.append(means[1])
            y2.append(means[2])
        patho = 'ARDS'
        plt.plot(pred_threshes, y1, label='{} sensitivity'.format(patho), lw=2)
        plt.plot(pred_threshes, y2, label='{} specificity'.format(patho), lw=2)
        plt.legend(loc='lower right')
        plt.title('Sensitivity v Specificity analysis')
        plt.ylabel('Score')
        plt.xlabel('Percentage ARDS votes')
        plt.ylim(0.0, 1.01)
        plt.yticks(np.arange(0, 1.01, .1))
        plt.xticks(np.arange(0, 101, 10))
        plt.grid()
        plt.show()

    def get_youdens_results(self):
        """
        Get Youden results for all models derived
        """
        results = self.get_all_patient_results_dataframe()
        uniq_pts = len(results.patient_id.unique())
        # -1 stands for the ground truth idx, and 2 stands for prediction frac idx
        all_tpr, all_fpr, threshs = janky_roc(results.ground_truth, results.frac_votes)
        j_scores = np.array(all_tpr) - np.array(all_fpr)
        tmp = zip(j_scores, threshs)
        ordered_j_scores = []
        for score, thresh in tmp:
            if thresh in np.arange(0, 101, 1) / 100.0:
                ordered_j_scores.append((score, thresh))
        ordered_j_scores = sorted(ordered_j_scores, key=lambda x: (x[0], -x[1]))
        optimal_pred_frac = ordered_j_scores[-1][1]
        data_at_frac = self.get_aggregate_predictions_dataframe(optimal_pred_frac)
        # get closest prediction thresh
        optimal_table = PrettyTable()
        optimal_table.field_names = ['patho', '% votes', 'acc', 'sen', 'spec', 'prec', 'npv']
        for patho in ['other', 'ards']:
            stats = self.get_summary_statistics_from_frame(data_at_frac, patho, optimal_pred_frac)
            means = stats.mean().round(2)
            cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
            optimal_table.add_row([
                patho,
                optimal_pred_frac,
                u"{}\u00B1{}".format(means[0], cis[0]),
                u"{}\u00B1{}".format(means[1], cis[1]),
                u"{}\u00B1{}".format(means[2], cis[2]),
                u"{}\u00B1{}".format(means[3], cis[3]),
                u"{}\u00B1{}".format(means[4], cis[4]),
            ])

        print('---Youden Results---')
        print(optimal_table)

    def get_summary_statistics_from_frame(self, dataframe, patho):
        """
        Get summary statistics about all models in question given a pathophysiology and
        threshold to evaluate at.
        """
        # XXX this is from old code in ardsdetection. but I'm just keeping it
        # around for a sec
        threshold = .5
        tps = "{}_tps_{}".format(patho, threshold)
        tns = "{}_tns_{}".format(patho, threshold)
        fps = "{}_fps_{}".format(patho, threshold)
        fns = "{}_fns_{}".format(patho, threshold)
        sens = dataframe[tps] / (dataframe[tps] + dataframe[fns])
        specs = dataframe[tns] / (dataframe[tns] + dataframe[fps])
        precs = dataframe[tps] / (dataframe[fps] + dataframe[tps])
        npvs = dataframe[tns] / (dataframe[tns] + dataframe[fns])
        accs = (dataframe[tns] + dataframe[tps]) / (dataframe[tns] + dataframe[tps] + dataframe[fns] + dataframe[fps])
        stats = pd.concat([accs, sens, specs, precs, npvs], axis=1)
        return stats

    def get_auc_results(self, patient_results):
        # XXX cant group in this func
        group = patient_results.groupby('epoch')
        aucs = []
        for i, model_pts in group:
            fpr, tpr, thresholds = roc_curve(model_pts.ground_truth, model_pts.frac_votes, pos_label=1)
            aucs.append(auc(fpr, tpr))
        return np.array(aucs)

    def print_thresh_table(self, thresh_interval):
        assert 1 <= thresh_interval <= 100
        table = PrettyTable()
        table.field_names = ['patho', 'vote %', 'acc', 'sen', 'spec', 'prec', 'npv']
        pred_threshes = range(0, 100+thresh_interval, thresh_interval)
        patient_results = self.get_all_patient_results_dataframe()
        uniq_pts = len(patient_results.patient_id.unique())
        for i in pred_threshes:
            thresh = i / 100.0
            df = self.get_aggregate_predictions_dataframe(thresh)
            stats = self.get_summary_statistics_from_frame(df, 'ards', thresh)
            means = stats.mean().round(2)
            cis = (1.96 * np.sqrt(means*(1-means)/uniq_pts)).round(3)
            row = [
                'ards',
                i,
                u"{}\u00B1{}".format(means[0], cis[0]),
                u"{}\u00B1{}".format(means[1], cis[1]),
                u"{}\u00B1{}".format(means[2], cis[2]),
                u"{}\u00B1{}".format(means[3], cis[3]),
                u"{}\u00B1{}".format(means[4], cis[5]),
            ]
            table.add_row(row)
        print(table)
