"""Experimental single-reviewer reward calibration; opt-in, never the default.

Linear utility and a three-outcome symmetric pairwise likelihood with ties.
Only the tie logit is an intercept: there is no learned preference for screen side.
"""
import math
import numpy as np
from .rendering import luminance, rgb
from .rewards import evaluate, weighted_quality

FEATURES = ('button_width', 'neighbor_luminance_difference', 'neighbor_center_distance', 'competing_text_size')
CHOICES = ('left', 'right', 'tie')


def features(state):
    ctas = [e for e in state.elements if e.type == 'shape' and e.role == 'cta']
    if len(ctas) != 1:
        raise ValueError('Feature model requires exactly one CTA')
    cta = ctas[0]
    neighbors = [e for e in state.elements if e.id != cta.id and e.color
                 and e.y+e.height <= cta.y]
    if not neighbors:
        raise ValueError('Feature model requires a filled element above the CTA')
    def center_distance(e):
        return math.hypot((cta.x+cta.width/2)-(e.x+e.width/2),
                          (cta.y+cta.height/2)-(e.y+e.height/2))
    neighbor = min(neighbors, key=lambda e:(center_distance(e),e.id))
    competing = [e for e in state.elements if e.type=='text' and e.role!='headline' and e.content]
    return np.array([cta.width/state.width,
        abs(float(luminance(rgb(cta.color)))-float(luminance(rgb(neighbor.color)))),
        center_distance(neighbor)/math.hypot(state.width,state.height),
        max((e.font_size for e in competing),default=0)/state.height],dtype=float)


def fit(left, right, labels, ridge=.1):
    left,right=np.asarray(left,float),np.asarray(right,float)
    if left.ndim!=2 or left.shape!=right.shape or left.shape[1]!=len(FEATURES) or len(left)!=len(labels) or not len(left):
        raise ValueError('Expected nonempty paired feature matrices')
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not math.isfinite(ridge) or ridge<=0:
        raise ValueError('Finite features and positive regularization required')
    y=np.array([CHOICES.index(label) for label in labels])
    delta=left-right
    scale=np.sqrt(np.mean(delta**2,axis=0))
    active=scale>1e-10
    # Never infer effects from between-pair variation: only within-pair differences.
    scale=np.where(active,scale,1.)
    x=delta[:,active]/scale[active]
    if np.linalg.matrix_rank(x)<x.shape[1]:
        raise ValueError('Active feature effects are confounded; simplify features before fitting')
    k=x.shape[1]
    design=np.zeros((len(x),3,k+1))
    design[:,0,:k]=x/2; design[:,1,:k]=-x/2; design[:,2,k]=1
    theta=np.zeros(k+1)
    def objective(t,derivatives=False):
        logits=design@t
        shift=logits.max(axis=1,keepdims=True)
        logz=shift[:,0]+np.log(np.exp(logits-shift).sum(axis=1))
        loss=float(np.mean(logz-logits[np.arange(len(y)),y])+ridge/2*(t@t))
        if not derivatives: return loss
        p=np.exp(logits-logz[:,None])
        mean=np.einsum('nc,nck->nk',p,design)
        grad=np.mean(mean-design[np.arange(len(y)),y],axis=0)+ridge*t
        hess=(np.einsum('nc,nci,ncj->ij',p,design,design)-mean.T@mean)/len(y)+ridge*np.eye(k+1)
        return loss,grad,hess
    for iteration in range(100):
        loss,grad,hess=objective(theta,True)
        if np.max(np.abs(grad))<1e-9: break
        step=np.linalg.solve(hess,grad)
        rate=1.
        while rate>1e-12 and objective(theta-rate*step)>loss-1e-4*rate*(grad@step): rate/=2
        if rate<=1e-12: raise ValueError('Optimization line search failed')
        theta-=rate*step
    else: raise ValueError('Optimization did not converge')
    weights=np.zeros(len(FEATURES)); weights[active]=theta[:k]/scale[active]
    combined=np.concatenate([left,right])
    return dict(version='local_context_fit_v1',features=list(FEATURES),weights=weights.tolist(),
        standardized_weights={name:float(weights[i]*scale[i]) for i,name in enumerate(FEATURES)},
        unidentified=[FEATURES[i] for i in range(len(FEATURES)) if not active[i]],
        tie_logit=float(theta[-1]),center=combined.mean(axis=0).tolist(),
        training_min=combined.min(axis=0).tolist(),training_max=combined.max(axis=0).tolist(),
        scale=scale.tolist(),ridge=ridge,training_pairs=len(left),iterations=iteration,
        gradient_max=float(np.max(np.abs(grad))),penalized_loss=loss,
        decision_rule='Largest class probability; exact symmetry of left/right maxima returns tie without a side preference.')


def predict(left,right,model):
    difference=float((np.asarray(left)-np.asarray(right))@np.asarray(model['weights']))
    logits=np.array([difference/2,-difference/2,model['tie_logit']])
    p=np.exp(logits-logits.max()); p/=p.sum()
    winners=np.flatnonzero(np.isclose(p,p.max(),rtol=0,atol=1e-10))
    choice=CHOICES[winners[0]] if len(winners)==1 else 'tie'
    return dict(choice=choice,probabilities=dict(zip(CHOICES,p.tolist())),utility_difference=difference)


def evaluate_fitted(state, model, alpha=.15):
    """Bounded optional reward; guard invalid designs with the existing validity cap.

    Absolute origin and mixing weight are conventions, not learned from pair choices.
    Out-of-range and unsupported states are explicitly flagged.
    """
    if not math.isfinite(alpha) or not 0<=alpha<=1: raise ValueError('Invalid mixing weight')
    baseline=evaluate(state)
    try:
        f=features(state)
    except ValueError as exc:
        return dict(score=baseline['score'],baseline_score=baseline['score'],
                    supported=False,reason=str(exc),experimental=True)
    u=float((f-np.asarray(model['center']))@np.asarray(model['weights']))
    qfit=float(1/(1+np.exp(-np.clip(u,-700,700))))
    q=(1-alpha)*weighted_quality(baseline['components'])+alpha*qfit
    if not baseline['essential_constraints_pass']: q=min(q,.49)
    outside=(f<np.asarray(model['training_min'])-1e-10)|(f>np.asarray(model['training_max'])+1e-10)
    return dict(score=round(2*q-1,6),baseline_score=baseline['score'],
        fitted_component=qfit,utility=u,alpha=alpha,supported=True,experimental=True,
        essential_constraints_pass=baseline['essential_constraints_pass'],
        outside_training_range=[FEATURES[i] for i,v in enumerate(outside) if v])
