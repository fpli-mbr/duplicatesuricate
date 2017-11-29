import pandas as pd

from duplicatesuricate.deduplication import RecordLinker,Launcher
from duplicatesuricate.evaluation import FuncEvaluationModel

if __name__ == '__main__':

    ### Load the reference records and the file to be checked as pd.DataFrame
    df_input_records = pd.read_excel('helicopters.xlsx',index_col=0,sheet_name='source')
    df_target_records = pd.read_excel('helicopters.xlsx',index_col=0,sheet_name='target')

    ## Create evaluation model
    loc_cols = ['location', 'constructor']
    id_cols = ['serieid']
    filter_dict = {'all': loc_cols, 'any': id_cols}
    intermediate_threshold = {'productname_fuzzyscore': 0.8}

    hard_threshold = {'location_fuzzyscore': 0.9,
                      'serieid_exactscore': 1}

    hard_cols = list(hard_threshold.keys())


    def hardcodedfunc(r):
        r = r.fillna(0)
        for k in hard_cols:
            if r[k] < hard_threshold[k]:
                return 0
        else:
            return 1

    model = FuncEvaluationModel(used_cols=hard_cols,
                                eval_func=hardcodedfunc)
    #optional
    model.fit()

    # create linker and launcher
    rl = RecordLinker(df=df_target_records,
                      filterdict=filter_dict,
                      intermediate_thresholds=intermediate_threshold,
                      evaluator=model)
    lh = Launcher(input_records=df_input_records,
                  target_records=df_target_records,
                  cleanfunc=None,
                  linker=rl)
    # Start linkage
    y = lh.start_linkage()
    z=lh.format_results(y,display=['productname','location','constructor'],fuzzy=['productname'])
    print(z)

